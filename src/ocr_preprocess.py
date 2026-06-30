from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


OCR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OCR_ROOT.parent
if str(OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_ROOT))

from benchmark_ocr import (  # noqa: E402
    DEFAULT_OCR_DIR,
    DEFAULT_PPOCR_DET_MODEL,
    DEFAULT_PPOCR_DEVICE,
    DEFAULT_PPOCR_REC_MODEL,
    IMAGE_EXTS,
    Prediction,
    add_ocr_paths,
    create_ocr_runner,
    normalize_group_id,
    normalize_image_id,
    normalize_model_names as normalize_benchmark_model_names,
    optional_progress,
    parse_ocr_result,
    patch_text_recognizer,
    resolve_ocr_module_dir,
)


DEFAULT_KEYFRAME_ROOT = OCR_ROOT / "data" / "raw"
DEFAULT_MAP_KEYFRAME_ROOT = PROJECT_ROOT / "ProcessedData" / "data" / "map_keyframes"
DEFAULT_FILES_LIST_ROOT = PROJECT_ROOT / "dataraw" / "folder_file_list"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ProcessedData" / "data" / "ocr"
MODEL_PPOCRV5_VIETOCR = "ppocrv5_vietocr"
HYBRID_MODEL_ALIASES = {
    "hybrid": MODEL_PPOCRV5_VIETOCR,
    "ppocr_vietocr": MODEL_PPOCRV5_VIETOCR,
    "ppocr-vietocr": MODEL_PPOCRV5_VIETOCR,
    "ppocrv5_vietocr": MODEL_PPOCRV5_VIETOCR,
    "ppocrv5-vietocr": MODEL_PPOCRV5_VIETOCR,
    "pp_ocrv5_vietocr": MODEL_PPOCRV5_VIETOCR,
    "pp-ocrv5-vietocr": MODEL_PPOCRV5_VIETOCR,
    "pp-ocrv5+vietocr": MODEL_PPOCRV5_VIETOCR,
}
PATH_FIELDS = (
    "image_path",
    "path",
    "original_path",
    "source_path",
    "keyframe_path",
    "frame_path",
)
GROUP_FIELDS = ("group_id", "video_id", "folder", "video", "clip_id")
IMAGE_ID_FIELDS = ("image_id", "frame_id", "frame", "idx", "filename", "name")
MAP_ID_FIELDS = ("id", "keyframe_id", "kf_id", "frame_idx")
SELECTED_FIELDS = ("selected", "is_selected", "keep", "is_keyframe", "keyframe")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class KeyframeItem:
    group_id: str
    image_id: str
    source_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.group_id, self.image_id)


@dataclass(frozen=True)
class PreparedPath:
    item: KeyframeItem
    ocr_path: Path
    preprocessed_path: Path | None


def first_present(row: Mapping[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "keep", "kept", "selected", "keyframe"}:
        return True
    if text in {"0", "false", "no", "n", "drop", "dropped", "discard"}:
        return False
    return None


def selected_by_manifest(row: Mapping[str, Any]) -> bool:
    for field_name in SELECTED_FIELDS:
        if field_name in row:
            selected = parse_bool(row.get(field_name))
            if selected is not None:
                return selected
    return True


def read_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("keyframes", "frames", "selected_keyframes", "items", "data", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        raise ValueError(f"Unsupported JSON manifest shape: {path}")
    raise ValueError(f"Unsupported manifest type: {path}. Use CSV, JSON, or JSONL.")


def path_candidates(raw_path: str, manifest_dir: Path, image_root: Path | None) -> Iterable[Path]:
    value = raw_path.strip().strip('"')
    if not value:
        return

    candidate = Path(value)
    if candidate.is_absolute():
        yield candidate
    else:
        yield manifest_dir / candidate
        if image_root is not None:
            yield image_root / candidate
        yield OCR_ROOT / candidate
        yield PROJECT_ROOT / candidate

    if candidate.suffix:
        return

    for ext in sorted(IMAGE_EXTS):
        if candidate.is_absolute():
            yield candidate.with_suffix(ext)
        else:
            yield (manifest_dir / candidate).with_suffix(ext)
            if image_root is not None:
                yield (image_root / candidate).with_suffix(ext)
            yield (OCR_ROOT / candidate).with_suffix(ext)
            yield (PROJECT_ROOT / candidate).with_suffix(ext)


def resolve_image_path(
    row: Mapping[str, Any],
    manifest_dir: Path,
    image_root: Path | None,
) -> Path | None:
    raw_path = first_present(row, PATH_FIELDS)
    if raw_path:
        for candidate in path_candidates(raw_path, manifest_dir, image_root):
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

    group_id = normalize_group_id(first_present(row, GROUP_FIELDS))
    image_id = normalize_image_id(first_present(row, IMAGE_ID_FIELDS))
    if image_root is None or not group_id or not image_id:
        return None

    for ext in sorted(IMAGE_EXTS):
        candidate = image_root / group_id / f"{image_id}{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def infer_group_id(path: Path, image_root: Path | None, row: Mapping[str, Any]) -> str:
    group_id = first_present(row, GROUP_FIELDS)
    if group_id:
        return normalize_group_id(group_id)
    if image_root is not None:
        try:
            return normalize_group_id(path.parent.relative_to(image_root).as_posix())
        except ValueError:
            pass
    return normalize_group_id(path.parent.name)


def infer_image_id(path: Path, row: Mapping[str, Any]) -> str:
    image_id = first_present(row, IMAGE_ID_FIELDS)
    if image_id:
        return normalize_image_id(image_id)
    return normalize_image_id(path.stem)


def keyframe_from_record(
    row: Mapping[str, Any],
    manifest_dir: Path,
    image_root: Path | None,
) -> KeyframeItem | None:
    path = resolve_image_path(row, manifest_dir, image_root)
    if path is None:
        return None
    group_id = infer_group_id(path, image_root, row)
    image_id = infer_image_id(path, row)
    if not group_id or not image_id:
        return None
    return KeyframeItem(group_id=group_id, image_id=image_id, source_path=path, metadata=dict(row))


def wait_for_existing_file(path: Path, timeout_seconds: float) -> bool:
    if path.exists():
        return True
    if timeout_seconds <= 0:
        return False

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(1.0)
    return path.exists()


def resolve_list_entry(base_dir: Path, raw_value: str, with_extension: str) -> Path:
    value = raw_value.strip().strip('"')
    path = Path(value)
    if with_extension and path.suffix.lower() != with_extension.lower():
        path = path.with_suffix(with_extension)
    if path.is_absolute():
        return path
    return base_dir / path


def load_files_list_paths(base_dir: Path, files_list_path: Path, with_extension: str) -> list[Path]:
    if not files_list_path.exists():
        return []
    with files_list_path.open("r", encoding="utf-8-sig") as f:
        return [
            resolve_list_entry(base_dir, line, with_extension)
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def read_map_keyframe_rows(map_path: Path) -> list[tuple[str, dict[str, Any]]]:
    with map_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or []
        id_field = next((field for field in MAP_ID_FIELDS if field in fieldnames), None)
        if id_field is None:
            raise ValueError(f"{map_path} must contain one of these id columns: {', '.join(MAP_ID_FIELDS)}")

        rows: list[tuple[str, dict[str, Any]]] = []
        for row in reader:
            raw_id = str(row.get(id_field, "")).strip()
            if raw_id:
                rows.append((raw_id, dict(row)))
        return rows


def frame_stem_from_map_id(raw_id: str) -> str:
    value = str(raw_id).strip()
    try:
        return f"{int(value):06d}"
    except ValueError:
        try:
            numeric_value = float(value)
        except ValueError:
            numeric_value = None
        if numeric_value is not None and numeric_value.is_integer():
            return f"{int(numeric_value):06d}"
        return normalize_image_id(value)


def find_keyframe_image(video_dir: Path, raw_id: str) -> Path | None:
    primary_stem = frame_stem_from_map_id(raw_id)
    fallback_stem = normalize_image_id(raw_id)
    stems = [primary_stem]
    if fallback_stem and fallback_stem not in stems:
        stems.append(fallback_stem)

    extensions = [".webp", *sorted(ext for ext in IMAGE_EXTS if ext != ".webp")]
    for stem in stems:
        for ext in extensions:
            candidate = video_dir / f"{stem}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

    for stem in stems:
        matches = sorted(path for path in video_dir.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTS)
        if matches:
            return matches[0].resolve()
    return None


def discover_subfolders(
    keyframe_root: Path,
    files_list_root: Path,
    requested_subfolders: list[str] | None,
) -> list[str]:
    if requested_subfolders:
        return sorted(dict.fromkeys(requested_subfolders))

    subfolders: set[str] = set()
    if keyframe_root.exists():
        subfolders.update(path.name for path in keyframe_root.iterdir() if path.is_dir())
    if files_list_root.exists():
        for path in files_list_root.glob("files_list_*.txt"):
            subfolders.add(path.stem.replace("files_list_", "", 1))
    return sorted(subfolders)


def keyframe_items_from_video_dir(video_dir: Path, keyframe_root: Path, subfolder: str | None) -> list[KeyframeItem]:
    items: list[KeyframeItem] = []
    for path in sorted(item for item in video_dir.iterdir() if item.suffix.lower() in IMAGE_EXTS):
        metadata = {"subfolder": subfolder or "", "source": "directory"}
        items.append(
            KeyframeItem(
                group_id=normalize_group_id(video_dir.name),
                image_id=normalize_image_id(path.stem),
                source_path=path.resolve(),
                metadata=metadata,
            )
        )
    return items


def discover_kf_embedding_keyframes(
    keyframe_root: Path,
    map_keyframe_root: Path,
    files_list_root: Path,
    requested_subfolders: list[str] | None,
    wait_map_timeout: float,
    allow_missing_map: bool,
) -> list[KeyframeItem]:
    if not keyframe_root.exists():
        raise FileNotFoundError(f"Keyframe root not found: {keyframe_root}")

    items: list[KeyframeItem] = []
    subfolders = discover_subfolders(keyframe_root, files_list_root, requested_subfolders)
    if not subfolders:
        raise FileNotFoundError(f"No keyframe subfolders found under: {keyframe_root}")

    for subfolder in subfolders:
        sub_keyframe_root = keyframe_root / subfolder
        sub_map_root = map_keyframe_root / subfolder
        files_list_path = files_list_root / f"files_list_{subfolder}.txt"

        if files_list_path.exists():
            video_dirs = load_files_list_paths(sub_keyframe_root, files_list_path, "")
            map_paths = load_files_list_paths(sub_map_root, files_list_path, ".csv")
        elif sub_keyframe_root.exists():
            video_dirs = sorted(path for path in sub_keyframe_root.iterdir() if path.is_dir())
            map_paths = [sub_map_root / f"{video_dir.name}.csv" for video_dir in video_dirs]
        else:
            continue

        for video_dir, map_path in zip(video_dirs, map_paths):
            if not video_dir.exists() or not video_dir.is_dir():
                continue
            if not wait_for_existing_file(map_path, wait_map_timeout):
                if allow_missing_map:
                    items.extend(keyframe_items_from_video_dir(video_dir, keyframe_root, subfolder))
                    continue
                raise FileNotFoundError(
                    f"Map keyframe CSV not found for {video_dir.name}: {map_path}. "
                    "This OCR step should run after kf-embedding/map_keyframes are ready."
                )

            for raw_id, row in read_map_keyframe_rows(map_path):
                image_path = find_keyframe_image(video_dir, raw_id)
                if image_path is None:
                    raise FileNotFoundError(
                        f"Could not find keyframe image for id={raw_id!r} under {video_dir}. "
                        "Expected names like 000001.webp from the kf-embedding loader."
                    )
                metadata = dict(row)
                metadata.update(
                    {
                        "map_id": raw_id,
                        "map_path": str(map_path),
                        "subfolder": subfolder,
                        "source": "kf_embedding",
                    }
                )
                items.append(
                    KeyframeItem(
                        group_id=normalize_group_id(video_dir.name),
                        image_id=normalize_image_id(image_path.stem),
                        source_path=image_path,
                        metadata=metadata,
                    )
                )
    return items


def discover_recursive_keyframes(keyframe_root: Path) -> list[KeyframeItem]:
    if not keyframe_root.exists():
        raise FileNotFoundError(f"Keyframe root not found: {keyframe_root}")

    root = keyframe_root.resolve()
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    items: list[KeyframeItem] = []
    for path in paths:
        group_id = normalize_group_id(path.parent.relative_to(root).as_posix())
        items.append(
            KeyframeItem(
                group_id=group_id,
                image_id=normalize_image_id(path.stem),
                source_path=path.resolve(),
                metadata={"source": "recursive"},
            )
        )
    return items


def discover_keyframes(
    keyframe_root: Path,
    manifest: Path | None,
    image_root: Path | None,
    limit: int | None,
    respect_selected_column: bool,
    layout: str,
    map_keyframe_root: Path,
    files_list_root: Path,
    subfolders: list[str] | None,
    wait_map_timeout: float,
    allow_missing_map: bool,
) -> list[KeyframeItem]:
    items: list[KeyframeItem] = []
    if manifest is not None:
        rows = read_manifest(manifest)
        manifest_dir = manifest.resolve().parent
        root = image_root.resolve() if image_root is not None and image_root.exists() else image_root
        for row in rows:
            if respect_selected_column and not selected_by_manifest(row):
                continue
            item = keyframe_from_record(row, manifest_dir, root)
            if item is not None:
                items.append(item)
    else:
        selected_layout = layout
        if selected_layout == "auto":
            try:
                using_default_raw = keyframe_root.resolve() == DEFAULT_KEYFRAME_ROOT.resolve()
            except OSError:
                using_default_raw = keyframe_root == DEFAULT_KEYFRAME_ROOT
            selected_layout = "recursive" if using_default_raw else (
                "kf-embedding" if map_keyframe_root.exists() else "recursive"
            )

        if selected_layout == "kf-embedding":
            items = discover_kf_embedding_keyframes(
                keyframe_root=keyframe_root,
                map_keyframe_root=map_keyframe_root,
                files_list_root=files_list_root,
                requested_subfolders=subfolders,
                wait_map_timeout=wait_map_timeout,
                allow_missing_map=allow_missing_map,
            )
        elif selected_layout == "recursive":
            items = discover_recursive_keyframes(keyframe_root)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

    deduped: dict[tuple[str, str], KeyframeItem] = {}
    for item in items:
        deduped.setdefault(item.key, item)

    result = [deduped[key] for key in sorted(deduped)]
    if limit is not None:
        result = result[:limit]
    if not result:
        source = manifest if manifest is not None else keyframe_root
        raise FileNotFoundError(f"No selected keyframe images found from: {source}")
    return result


def output_image_path(item: KeyframeItem, output_root: Path, image_format: str) -> Path:
    suffix = ".png" if image_format.lower() == "png" else ".jpg"
    return output_root / "preprocessed" / item.group_id / f"{item.image_id}{suffix}"


def load_original_image(path: Path) -> Any:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError("Install Pillow before running OCR preprocessing.") from exc

    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_for_ocr(image: Any, max_side: int, min_side: int) -> Any:
    from PIL import Image

    if max_side <= 0 and min_side <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    shortest = min(width, height)
    scale = 1.0
    max_scale = max_side / float(longest) if max_side > 0 else float("inf")
    if max_side > 0 and longest > max_side:
        scale = min(scale, max_scale)
    if min_side > 0 and shortest < min_side:
        upscale = min_side / float(shortest)
        if max_side > 0:
            upscale = min(upscale, max_scale)
        scale = max(scale, upscale)
    if scale == 1.0:
        return image

    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return image.resize(new_size, resample)


def preprocess_image(image: Any, args: argparse.Namespace) -> Any:
    from PIL import ImageEnhance, ImageOps

    max_side = args.det_max_side if args.det_max_side is not None else args.max_side
    min_side = args.det_min_side if args.det_min_side is not None else args.min_side
    autocontrast = args.det_autocontrast if args.det_autocontrast is not None else args.autocontrast
    autocontrast_cutoff = (
        args.det_autocontrast_cutoff
        if args.det_autocontrast_cutoff is not None
        else args.autocontrast_cutoff
    )
    contrast = args.det_contrast if args.det_contrast is not None else args.contrast
    sharpness = args.det_sharpness if args.det_sharpness is not None else args.sharpness

    processed = resize_for_ocr(image, max_side, min_side)
    if args.grayscale:
        processed = processed.convert("L").convert("RGB")
    if autocontrast:
        processed = ImageOps.autocontrast(processed, cutoff=autocontrast_cutoff)
    if contrast != 1.0:
        processed = ImageEnhance.Contrast(processed).enhance(contrast)
    if sharpness != 1.0:
        processed = ImageEnhance.Sharpness(processed).enhance(sharpness)
    return processed


def order_points_clockwise(points: Any) -> Any:
    import numpy as np

    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    point_sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(point_sum)]
    rect[2] = pts[np.argmax(point_sum)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def clip_box(points: Any, width: int, height: int) -> Any:
    import numpy as np

    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
    return pts


def box_size(points: Any) -> tuple[float, float]:
    import numpy as np

    pts = order_points_clockwise(points)
    box_width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    box_height = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    return float(box_width), float(box_height)


def sort_text_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not boxes:
        return []

    heights = [box_size(box["box"])[1] for box in boxes]
    median_height = sorted(heights)[len(heights) // 2] if heights else 10.0
    same_line_threshold = max(10.0, median_height * 0.5)

    ordered = sorted(boxes, key=lambda item: (order_points_clockwise(item["box"])[0][1], order_points_clockwise(item["box"])[0][0]))
    for idx in range(len(ordered) - 1):
        for inner in range(idx, -1, -1):
            current = order_points_clockwise(ordered[inner]["box"])
            nxt = order_points_clockwise(ordered[inner + 1]["box"])
            if abs(float(nxt[0][1] - current[0][1])) < same_line_threshold and nxt[0][0] < current[0][0]:
                ordered[inner], ordered[inner + 1] = ordered[inner + 1], ordered[inner]
            else:
                break
    return ordered


def perspective_crop(image_rgb: Any, box: Any) -> Any:
    import cv2
    import numpy as np

    points = order_points_clockwise(box)
    crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    crop_width = max(1, crop_width)
    crop_height = max(1, crop_height)
    target = np.float32([[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(
        image_rgb,
        matrix,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] / max(1, crop.shape[1]) >= 1.5:
        crop = np.rot90(crop)
    return crop


def preprocess_crop_for_vietocr(crop_rgb: Any, args: argparse.Namespace) -> Any:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps

    crop = np.asarray(crop_rgb, dtype="uint8")
    padding = max(0, int(args.crop_padding))
    if padding:
        crop = cv2.copyMakeBorder(crop, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    target_height = int(args.crop_height)
    if target_height > 0 and crop.shape[0] > 0:
        scale = target_height / float(crop.shape[0])
        new_width = max(1, int(round(crop.shape[1] * scale)))
        crop = cv2.resize(crop, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    image = Image.fromarray(crop).convert("RGB")
    if args.crop_autocontrast:
        image = ImageOps.autocontrast(image, cutoff=args.crop_autocontrast_cutoff)
    if args.crop_contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(args.crop_contrast)
    if args.crop_sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(args.crop_sharpness)
    return np.array(image)


def save_preprocessed_image(image: Any, path: Path, image_format: str, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format.lower() == "png":
        image.save(path, format="PNG", optimize=True)
    else:
        image.save(path, format="JPEG", quality=jpeg_quality, optimize=True)


def prepare_item(item: KeyframeItem, args: argparse.Namespace) -> tuple[Any, PreparedPath]:
    image = load_original_image(item.source_path)
    processed = preprocess_image(image, args)
    preprocessed_path = None
    ocr_path = item.source_path
    if args.save_preprocessed:
        preprocessed_path = output_image_path(item, args.output_root, args.preprocessed_format)
        save_preprocessed_image(processed, preprocessed_path, args.preprocessed_format, args.jpeg_quality)
        ocr_path = preprocessed_path
    return processed, PreparedPath(item=item, ocr_path=ocr_path, preprocessed_path=preprocessed_path)


def maybe_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for attr_name in ("res", "json"):
        if not hasattr(value, attr_name):
            continue
        attr_value = getattr(value, attr_name)
        if callable(attr_value):
            try:
                attr_value = attr_value()
            except TypeError:
                continue
        if isinstance(attr_value, Mapping):
            return attr_value
    if hasattr(value, "to_dict"):
        try:
            attr_value = value.to_dict()
        except TypeError:
            attr_value = None
        if isinstance(attr_value, Mapping):
            return attr_value
    return None


def sequence_like(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_at(scores: Any, index: int) -> float | None:
    if scores is None:
        return None
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if sequence_like(scores):
        return to_float(scores[index]) if index < len(scores) else None
    return to_float(scores)


def rect_to_quad(rect: Any) -> Any:
    import numpy as np

    x0, y0, x1, y1 = np.asarray(rect, dtype="float32").reshape(4)
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")


def box_to_quad(value: Any) -> Any | None:
    import numpy as np

    try:
        arr = np.asarray(value, dtype="float32")
    except (TypeError, ValueError):
        return None
    if arr.shape == (4, 2):
        return arr
    if arr.ndim == 1 and arr.size == 8:
        return arr.reshape(4, 2)
    if arr.ndim == 1 and arr.size == 4:
        return rect_to_quad(arr)
    return None


def add_box_records(records: list[dict[str, Any]], boxes: Any, scores: Any = None) -> bool:
    import numpy as np

    try:
        arr = np.asarray(boxes, dtype="float32")
    except (TypeError, ValueError):
        arr = None

    if arr is not None:
        if arr.ndim == 3 and arr.shape[1:] == (4, 2):
            for idx, box in enumerate(arr):
                records.append({"box": box, "det_confidence": score_at(scores, idx)})
            return True
        if arr.ndim == 2 and arr.shape[1] in {4, 8} and arr.shape[0] != 4:
            added = False
            for idx, box in enumerate(arr):
                quad = box_to_quad(box)
                if quad is not None:
                    records.append({"box": quad, "det_confidence": score_at(scores, idx)})
                    added = True
            return added

    quad = box_to_quad(boxes)
    if quad is not None:
        records.append({"box": quad, "det_confidence": score_at(scores, 0)})
        return True

    if sequence_like(boxes):
        added = False
        for idx, item in enumerate(boxes):
            quad = box_to_quad(item)
            if quad is not None:
                records.append({"box": quad, "det_confidence": score_at(scores, idx)})
                added = True
        return added
    return False


def extract_ppocr_detection_boxes(result: Any, image_shape: tuple[int, ...], min_box_side: float) -> list[dict[str, Any]]:
    import numpy as np

    records: list[dict[str, Any]] = []
    box_keys = (
        "dt_polys",
        "det_polys",
        "rec_polys",
        "polys",
        "dt_boxes",
        "det_boxes",
        "rec_boxes",
        "boxes",
    )
    score_keys = ("dt_scores", "det_scores", "scores", "confidence")

    def visit(value: Any) -> None:
        mapping = maybe_mapping(value)
        if mapping is not None:
            payload = mapping.get("res") if isinstance(mapping.get("res"), Mapping) else mapping
            for box_key in box_keys:
                if box_key not in payload or payload[box_key] is None:
                    continue
                score_value = next((payload[key] for key in score_keys if key in payload), None)
                if add_box_records(records, payload[box_key], score_value):
                    return
            for nested in payload.values():
                if nested is not value:
                    visit(nested)
            return

        if sequence_like(value):
            if add_box_records(records, value):
                return
            for item in value:
                visit(item)

    visit(result)

    height, width = int(image_shape[0]), int(image_shape[1])
    filtered: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for record in records:
        box = clip_box(record["box"], width, height)
        box_width, box_height = box_size(box)
        if box_width < min_box_side or box_height < min_box_side:
            continue
        key = tuple(np.round(box.reshape(-1)).astype(int).tolist())
        if key in seen:
            continue
        seen.add(key)
        filtered.append({"box": box, "det_confidence": record.get("det_confidence")})
    return sort_text_boxes(filtered)


class PPOCRV5Detector:
    def __init__(self, args: argparse.Namespace) -> None:
        self.min_box_side = float(args.det_min_box_side)
        self.mode = "text_detection"
        self._legacy_cls = args.ppocr_textline_orientation

        try:
            from paddleocr import TextDetection
        except ImportError:
            TextDetection = None

        if TextDetection is not None:
            base_kwargs: dict[str, Any] = {"model_name": args.ppocr_det_model}
            if args.ppocr_device:
                base_kwargs["device"] = args.ppocr_device
            if args.ppocr_det_model_dir:
                base_kwargs["model_dir"] = str(args.ppocr_det_model_dir)
            for kwargs in (base_kwargs, {"model_name": args.ppocr_det_model}, {}):
                try:
                    self.detector = TextDetection(**kwargs)
                    return
                except TypeError:
                    continue

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PP-OCRv5 detector requires PaddleOCR. Install it with: "
                "python -m pip install paddleocr paddlepaddle"
            ) from exc

        self.mode = "paddleocr"
        kwargs: dict[str, Any] = {
            "text_detection_model_name": args.ppocr_det_model,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": args.ppocr_textline_orientation,
            "enable_mkldnn": args.ppocr_enable_mkldnn,
            "cpu_threads": args.ppocr_cpu_threads,
        }
        if args.ppocr_lang:
            kwargs["lang"] = args.ppocr_lang
        if args.ppocr_device:
            kwargs["device"] = args.ppocr_device
        if args.ppocr_det_model_dir:
            kwargs["text_detection_model_dir"] = str(args.ppocr_det_model_dir)

        try:
            self.detector = PaddleOCR(**kwargs)
        except TypeError:
            legacy_kwargs: dict[str, Any] = {
                "lang": args.ppocr_lang or "ch",
                "use_angle_cls": args.ppocr_textline_orientation,
                "ocr_version": "PP-OCRv5",
            }
            if args.ppocr_device:
                legacy_kwargs["use_gpu"] = str(args.ppocr_device).lower().startswith("gpu")
            self.detector = PaddleOCR(**legacy_kwargs)

    def __call__(self, image_array: Any) -> list[dict[str, Any]]:
        if self.mode == "text_detection":
            try:
                result = self.detector.predict(image_array)
            except TypeError:
                result = self.detector.predict(input=image_array)
        else:
            if hasattr(self.detector, "ocr"):
                try:
                    result = self.detector.ocr(image_array, det=True, rec=False, cls=self._legacy_cls)
                except TypeError:
                    result = self.detector.ocr(image_array, cls=self._legacy_cls)
            else:
                result = self.detector.predict(image_array)
        return extract_ppocr_detection_boxes(result, image_array.shape, self.min_box_side)


def load_vietocr_recognizer(ocr_dir: Path) -> Any:
    module_dir = resolve_ocr_module_dir(ocr_dir)
    add_ocr_paths(module_dir)
    try:
        import module.ocr as ocr_module
    except ModuleNotFoundError as exc:
        if exc.name in {"module", "module.ocr"}:
            raise ImportError(f"Could not import module.ocr from {module_dir}") from exc
        raise

    patch_text_recognizer(ocr_module, module_dir)
    return ocr_module.TextRecognizer(model_dir=None, device_id=0)


def clean_text(value: Any) -> str:
    text = CONTROL_RE.sub(" ", str(value))
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def postprocess_ocr_lines(
    text_lines: list[str],
    all_lines: list[dict[str, Any]],
    dedupe_lines: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    cleaned_lines: list[str] = []
    seen: set[str] = set()
    for line in text_lines:
        cleaned = clean_text(line)
        if not cleaned:
            continue
        dedupe_key = cleaned.casefold()
        if dedupe_lines and dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned_lines.append(cleaned)

    cleaned_all_lines: list[dict[str, Any]] = []
    for line in all_lines:
        cleaned = clean_text(line.get("text", ""))
        if cleaned:
            cleaned_all_lines.append({"text": cleaned, "confidence": line.get("confidence")})
    return cleaned_lines, cleaned_all_lines


def postprocess_hybrid_line_records(
    line_records: list[dict[str, Any]],
    confidence_threshold: float,
    dedupe_lines: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    accepted: list[str] = []
    all_lines: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line in line_records:
        cleaned = clean_text(line.get("text", ""))
        if not cleaned:
            continue
        record = dict(line)
        record["text"] = cleaned
        all_lines.append(record)

        confidence = record.get("confidence")
        if confidence is not None and confidence < confidence_threshold:
            continue
        dedupe_key = cleaned.casefold()
        if dedupe_lines and dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        accepted.append(cleaned)
    return accepted, all_lines


def recognize_vietocr_crops(recognizer: Any, crops: list[Any]) -> list[tuple[str, float | None]]:
    if not crops:
        return []
    rec_result = recognizer(crops)
    if isinstance(rec_result, tuple):
        rec_result = rec_result[0]

    results: list[tuple[str, float | None]] = []
    for item in rec_result:
        if isinstance(item, (list, tuple)) and item:
            text = str(item[0])
            confidence = to_float(item[1]) if len(item) > 1 else None
        else:
            text = str(item)
            confidence = None
        results.append((text, confidence))
    return results


def run_ppocr_vietocr_model(
    items: list[KeyframeItem],
    args: argparse.Namespace,
) -> tuple[list[Prediction], dict[tuple[str, str], PreparedPath]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Install numpy before running OCR.") from exc

    detector = PPOCRV5Detector(args)
    recognizer = load_vietocr_recognizer(args.ocr_dir)
    predictions: list[Prediction] = []
    prepared_paths: dict[tuple[str, str], PreparedPath] = {}

    for item in optional_progress(items, len(items), desc=f"OCR {MODEL_PPOCRV5_VIETOCR}"):
        processed, prepared = prepare_item(item, args)
        prepared_paths[item.key] = prepared
        frame_rgb = np.asarray(processed, dtype="uint8")

        start = time.perf_counter()
        boxes = detector(frame_rgb)
        crops: list[Any] = []
        valid_boxes: list[dict[str, Any]] = []
        for box_record in boxes:
            crop = perspective_crop(frame_rgb, box_record["box"])
            if crop.size == 0:
                continue
            crops.append(preprocess_crop_for_vietocr(crop, args))
            valid_boxes.append(box_record)

        rec_results = recognize_vietocr_crops(recognizer, crops)
        latency_ms = (time.perf_counter() - start) * 1000.0

        line_records: list[dict[str, Any]] = []
        for line_idx, (box_record, rec_result) in enumerate(zip(valid_boxes, rec_results)):
            text, confidence = rec_result
            box = order_points_clockwise(box_record["box"])
            line_records.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "det_confidence": box_record.get("det_confidence"),
                    "box": box.tolist(),
                    "line_index": line_idx,
                }
            )

        text_lines, all_lines = postprocess_hybrid_line_records(
            line_records,
            confidence_threshold=args.confidence,
            dedupe_lines=args.dedupe_lines,
        )
        predictions.append(
            Prediction(
                group_id=item.group_id,
                image_id=item.image_id,
                path=str(prepared.ocr_path),
                text_lines=tuple(text_lines),
                latency_ms=latency_ms,
                all_lines=tuple(all_lines),
            )
        )
    return predictions, prepared_paths


def prediction_to_record(
    pred: Prediction,
    item: KeyframeItem,
    preprocessed_path: Path | None,
) -> dict[str, Any]:
    return {
        "group_id": pred.group_id,
        "image_id": pred.image_id,
        "video_id": pred.group_id,
        "frame_id": pred.image_id,
        "idx": pred.image_id,
        "map_id": item.metadata.get("map_id", ""),
        "subfolder": item.metadata.get("subfolder", ""),
        "path": pred.path,
        "original_path": str(item.source_path),
        "preprocessed_path": str(preprocessed_path) if preprocessed_path is not None else "",
        "text_lines": list(pred.text_lines),
        "text": pred.text,
        "line_count": len(pred.text_lines),
        "latency_ms": pred.latency_ms,
        "all_lines": list(pred.all_lines),
    }


def group_json_name(group_id: str) -> str:
    return group_id.replace("\\", "__").replace("/", "__")


def write_ocr_outputs(
    predictions: list[Prediction],
    prepared_paths: Mapping[tuple[str, str], PreparedPath],
    output_root: Path,
    write_video_json: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            prepared = prepared_paths[pred.key]
            record = prediction_to_record(pred, prepared.item, prepared.preprocessed_path)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = output_root / "ocr_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "group_id",
            "image_id",
            "video_id",
            "frame_id",
            "idx",
            "map_id",
            "subfolder",
            "path",
            "original_path",
            "preprocessed_path",
            "text",
            "line_count",
            "latency_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pred in predictions:
            prepared = prepared_paths[pred.key]
            writer.writerow(
                {
                    "group_id": pred.group_id,
                    "image_id": pred.image_id,
                    "video_id": pred.group_id,
                    "frame_id": pred.image_id,
                    "idx": pred.image_id,
                    "map_id": prepared.item.metadata.get("map_id", ""),
                    "subfolder": prepared.item.metadata.get("subfolder", ""),
                    "path": pred.path,
                    "original_path": str(prepared.item.source_path),
                    "preprocessed_path": (
                        str(prepared.preprocessed_path) if prepared.preprocessed_path is not None else ""
                    ),
                    "text": pred.text,
                    "line_count": len(pred.text_lines),
                    "latency_ms": pred.latency_ms,
                }
            )

    if not write_video_json:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for pred in predictions:
        prepared = prepared_paths[pred.key]
        grouped.setdefault(pred.group_id, []).append(
            {
                "idx": pred.image_id,
                "image_id": pred.image_id,
                "frame_id": pred.image_id,
                "map_id": prepared.item.metadata.get("map_id", ""),
                "subfolder": prepared.item.metadata.get("subfolder", ""),
                "path": str(prepared.item.source_path),
                "preprocessed_path": (
                    str(prepared.preprocessed_path) if prepared.preprocessed_path is not None else ""
                ),
                "text": list(pred.text_lines),
                "text_joined": pred.text,
                "latency_ms": pred.latency_ms,
                "all_lines": list(pred.all_lines),
            }
        )

    for group_id, records in grouped.items():
        records.sort(key=lambda row: normalize_image_id(row["image_id"]))
        with (output_root / f"{group_json_name(group_id)}.json").open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def write_keyframe_manifest(
    items: list[KeyframeItem],
    prepared_paths: Mapping[tuple[str, str], PreparedPath],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "selected_keyframes.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "group_id",
            "image_id",
            "video_id",
            "frame_id",
            "map_id",
            "subfolder",
            "original_path",
            "preprocessed_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            prepared = prepared_paths.get(item.key)
            writer.writerow(
                {
                    "group_id": item.group_id,
                    "image_id": item.image_id,
                    "video_id": item.group_id,
                    "frame_id": item.image_id,
                    "map_id": item.metadata.get("map_id", ""),
                    "subfolder": item.metadata.get("subfolder", ""),
                    "original_path": str(item.source_path),
                    "preprocessed_path": (
                        str(prepared.preprocessed_path)
                        if prepared is not None and prepared.preprocessed_path is not None
                        else ""
                    ),
                }
            )


def run_preprocessing(items: list[KeyframeItem], args: argparse.Namespace) -> dict[tuple[str, str], PreparedPath]:
    prepared_paths: dict[tuple[str, str], PreparedPath] = {}
    for item in optional_progress(items, len(items), desc="OCR preprocessing"):
        _, prepared = prepare_item(item, args)
        prepared_paths[item.key] = prepared
    return prepared_paths


def run_ocr_model(
    model_name: str,
    items: list[KeyframeItem],
    args: argparse.Namespace,
) -> tuple[list[Prediction], dict[tuple[str, str], PreparedPath]]:
    if model_name == MODEL_PPOCRV5_VIETOCR:
        return run_ppocr_vietocr_model(items, args)

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Install numpy before running OCR.") from exc

    engine = create_ocr_runner(model_name, args)
    predictions: list[Prediction] = []
    prepared_paths: dict[tuple[str, str], PreparedPath] = {}

    for item in optional_progress(items, len(items), desc=f"OCR {model_name}"):
        processed, prepared = prepare_item(item, args)
        prepared_paths[item.key] = prepared
        start = time.perf_counter()
        raw_result = engine(np.array(processed), prepared.ocr_path)
        latency_ms = (time.perf_counter() - start) * 1000.0
        text_lines, all_lines = parse_ocr_result(raw_result, args.confidence)
        text_lines, all_lines = postprocess_ocr_lines(text_lines, all_lines, args.dedupe_lines)
        predictions.append(
            Prediction(
                group_id=item.group_id,
                image_id=item.image_id,
                path=str(prepared.ocr_path),
                text_lines=tuple(text_lines),
                latency_ms=latency_ms,
                all_lines=tuple(all_lines),
            )
        )
    return predictions, prepared_paths


def normalize_pipeline_model_names(values: list[str] | None) -> list[str]:
    requested = values or [MODEL_PPOCRV5_VIETOCR]
    models: list[str] = []
    for value in requested:
        normalized = str(value).strip().lower().replace(" ", "_")
        model_name = HYBRID_MODEL_ALIASES.get(normalized)
        if model_name is not None:
            if model_name not in models:
                models.append(model_name)
            continue
        for benchmark_model in normalize_benchmark_model_names([value]):
            if benchmark_model not in models:
                models.append(benchmark_model)
    return models


def model_output_root(base_output_root: Path, model_name: str, multi_model: bool) -> Path:
    return base_output_root / model_name if multi_model else base_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess keyframes after kf-embedding for OCR, then optionally run OCR detection, "
            "recognition, postprocessing, and result export."
        )
    )
    parser.add_argument("--keyframe-root", type=Path, default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--map-keyframe-root", type=Path, default=DEFAULT_MAP_KEYFRAME_ROOT)
    parser.add_argument("--files-list-root", type=Path, default=DEFAULT_FILES_LIST_ROOT)
    parser.add_argument(
        "--layout",
        choices=["auto", "kf-embedding", "recursive"],
        default="auto",
        help=(
            "auto uses the kf-embedding layout when map_keyframes exists; "
            "kf-embedding reads map_keyframes/<subfolder>/<video>.csv id values; "
            "recursive scans every image under --keyframe-root."
        ),
    )
    parser.add_argument(
        "--subfolders",
        nargs="+",
        default=None,
        help="Optional subset such as L30 L31. Defaults to subfolders discovered from keyframes/files_list.",
    )
    parser.add_argument(
        "--wait-map-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for each map_keyframes CSV, useful when OCR starts right after kf-embedding.",
    )
    parser.add_argument(
        "--allow-missing-map",
        action="store_true",
        help="Fallback to all images in a video folder if its map_keyframes CSV is missing.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV/JSON/JSONL override. If omitted, the kf-embedding layout is used by default.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Root used to resolve group_id/image_id rows in the manifest. Defaults to --keyframe-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ignore-selected-column",
        action="store_true",
        help="Use every manifest row even when selected/is_keyframe/keep columns say false.",
    )

    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--min-side", type=int, default=720)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--autocontrast", action="store_true")
    parser.add_argument("--autocontrast-cutoff", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--det-max-side", type=int, default=None)
    parser.add_argument("--det-min-side", type=int, default=None)
    parser.add_argument("--det-autocontrast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--det-autocontrast-cutoff", type=float, default=1.0)
    parser.add_argument("--det-contrast", type=float, default=1.08)
    parser.add_argument("--det-sharpness", type=float, default=1.12)
    parser.add_argument("--det-min-box-side", type=float, default=3.0)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--crop-height", type=int, default=48)
    parser.add_argument("--crop-autocontrast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-autocontrast-cutoff", type=float, default=0.5)
    parser.add_argument("--crop-contrast", type=float, default=1.12)
    parser.add_argument("--crop-sharpness", type=float, default=1.18)
    parser.add_argument("--dedupe-lines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.add_argument("--preprocessed-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)

    parser.add_argument("--run-ocr", action="store_true")
    parser.add_argument("--models", nargs="+", default=[MODEL_PPOCRV5_VIETOCR])
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-per-video-json", action="store_true")

    parser.add_argument("--ppocr-det-model", default=DEFAULT_PPOCR_DET_MODEL)
    parser.add_argument("--ppocr-rec-model", default=DEFAULT_PPOCR_REC_MODEL)
    parser.add_argument("--ppocr-det-model-dir", type=Path, default=None)
    parser.add_argument("--ppocr-rec-model-dir", type=Path, default=None)
    parser.add_argument("--ppocr-device", default=DEFAULT_PPOCR_DEVICE)
    parser.add_argument("--ppocr-lang", default=None)
    parser.add_argument("--ppocr-cpu-threads", type=int, default=10)
    parser.add_argument("--ppocr-enable-mkldnn", action="store_true")
    parser.add_argument("--ppocr-textline-orientation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    image_root = args.image_root or args.keyframe_root
    if image_root is not None:
        image_root = image_root.resolve()

    items = discover_keyframes(
        keyframe_root=args.keyframe_root.resolve(),
        manifest=args.manifest.resolve() if args.manifest is not None else None,
        image_root=image_root,
        limit=args.limit,
        respect_selected_column=not args.ignore_selected_column,
        layout=args.layout,
        map_keyframe_root=args.map_keyframe_root.resolve(),
        files_list_root=args.files_list_root.resolve(),
        subfolders=args.subfolders,
        wait_map_timeout=args.wait_map_timeout,
        allow_missing_map=args.allow_missing_map,
    )
    print(f"[INFO] Selected keyframes: {len(items)}")

    models = normalize_pipeline_model_names(args.models)
    if not args.run_ocr:
        prepared_paths = run_preprocessing(items, args)
        write_keyframe_manifest(items, prepared_paths, args.output_root)
        print(f"[INFO] Preprocessing manifest saved to: {args.output_root / 'selected_keyframes.csv'}")
        if args.save_preprocessed:
            print(f"[INFO] Preprocessed images saved under: {args.output_root / 'preprocessed'}")
        else:
            print("[INFO] Use --save-preprocessed to persist preprocessed images.")
        return

    multi_model = len(models) > 1
    root_prepared_paths: dict[tuple[str, str], PreparedPath] = {}
    for model_name in models:
        output_root = model_output_root(args.output_root, model_name, multi_model)
        if not args.overwrite and (output_root / "predictions.jsonl").exists():
            print(f"[INFO] Skipping {model_name}; predictions already exist at {output_root}")
            continue
        print(f"[INFO] Running OCR model: {model_name}")
        predictions, prepared_paths = run_ocr_model(model_name, items, args)
        if not root_prepared_paths:
            root_prepared_paths = prepared_paths
        write_ocr_outputs(
            predictions,
            prepared_paths,
            output_root=output_root,
            write_video_json=not args.no_per_video_json,
        )
        print(f"[INFO] Saved {len(predictions)} OCR records to: {output_root}")

    if root_prepared_paths:
        write_keyframe_manifest(items, root_prepared_paths, args.output_root)
        print(f"[INFO] Keyframe manifest saved to: {args.output_root / 'selected_keyframes.csv'}")


if __name__ == "__main__":
    main()
