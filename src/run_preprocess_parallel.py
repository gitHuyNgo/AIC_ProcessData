from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PREPROCESS_DIR = ROOT / "preprocess_data"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

import ocr_preprocess as ocr_preprocess  # noqa: E402


WORKER_ARGS: argparse.Namespace | None = None


def init_worker(args: argparse.Namespace) -> None:
    global WORKER_ARGS
    WORKER_ARGS = args


def process_item(item: Any) -> tuple[tuple[str, str], Any]:
    if WORKER_ARGS is None:
        raise RuntimeError("Worker arguments were not initialized.")

    preprocessed_path = None
    if WORKER_ARGS.save_preprocessed:
        preprocessed_path = ocr_preprocess.output_image_path(
            item,
            WORKER_ARGS.output_root,
            WORKER_ARGS.preprocessed_format,
        )
        if WORKER_ARGS.skip_existing and preprocessed_path.exists() and preprocessed_path.stat().st_size > 0:
            prepared = ocr_preprocess.PreparedPath(
                item=item,
                ocr_path=preprocessed_path,
                preprocessed_path=preprocessed_path,
            )
            return item.key, prepared

    for attempt in range(WORKER_ARGS.retries + 1):
        try:
            _, prepared = ocr_preprocess.prepare_item(item, WORKER_ARGS)
            return item.key, prepared
        except PermissionError:
            if (
                preprocessed_path is not None
                and preprocessed_path.exists()
                and preprocessed_path.stat().st_size > 0
            ):
                prepared = ocr_preprocess.PreparedPath(
                    item=item,
                    ocr_path=preprocessed_path,
                    preprocessed_path=preprocessed_path,
                )
                return item.key, prepared
            if attempt >= WORKER_ARGS.retries:
                raise
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError("Unreachable retry state.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel runner for preprocess_data/ocr_preprocess.py preprocessing."
    )
    parser.add_argument("--keyframe-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("preprocessed_results"))
    parser.add_argument("--layout", choices=["auto", "kf-embedding", "recursive"], default="recursive")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true", default=True)

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
    parser.add_argument("--save-preprocessed", action="store_true", default=True)
    parser.add_argument("--preprocessed-format", choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.keyframe_root = args.keyframe_root.resolve()
    args.output_root = args.output_root.resolve()
    args.map_keyframe_root = ocr_preprocess.DEFAULT_MAP_KEYFRAME_ROOT.resolve()
    args.files_list_root = ocr_preprocess.DEFAULT_FILES_LIST_ROOT.resolve()
    args.subfolders = None
    args.wait_map_timeout = 0.0
    args.allow_missing_map = False
    args.ignore_selected_column = False

    image_root = args.image_root or args.keyframe_root
    if image_root is not None:
        image_root = image_root.resolve()

    items = ocr_preprocess.discover_keyframes(
        keyframe_root=args.keyframe_root,
        manifest=args.manifest.resolve() if args.manifest is not None else None,
        image_root=image_root,
        limit=args.limit,
        respect_selected_column=not args.ignore_selected_column,
        layout=args.layout,
        map_keyframe_root=args.map_keyframe_root,
        files_list_root=args.files_list_root,
        subfolders=args.subfolders,
        wait_map_timeout=args.wait_map_timeout,
        allow_missing_map=args.allow_missing_map,
    )
    print(f"[INFO] Selected keyframes: {len(items)}")
    print(f"[INFO] Workers: {args.workers}; chunksize: {args.chunksize}")

    prepared_paths: dict[tuple[str, str], Any] = {}
    started = time.perf_counter()
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args,)) as pool:
        iterator = pool.imap_unordered(process_item, items, chunksize=args.chunksize)
        for key, prepared in ocr_preprocess.optional_progress(
            iterator,
            total=len(items),
            desc="OCR preprocessing",
        ):
            prepared_paths[key] = prepared

    ocr_preprocess.write_keyframe_manifest(items, prepared_paths, args.output_root)
    elapsed = time.perf_counter() - started
    print(f"[INFO] Preprocessing manifest saved to: {args.output_root / 'selected_keyframes.csv'}")
    print(f"[INFO] Preprocessed images saved under: {args.output_root / 'preprocessed'}")
    print(f"[INFO] Done in {elapsed / 60.0:.2f} minutes")


if __name__ == "__main__":
    mp.freeze_support()
    main()
