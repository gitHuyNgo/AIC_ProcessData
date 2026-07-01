from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / \
    "video_processing" if SCRIPT_DIR.name == "final" else SCRIPT_DIR
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect scene boundaries with PySceneDetect ContentDetector, saving old TransNet-style scene txt files."
    )
    parser.add_argument("--content-threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=int, default=15)
    parser.add_argument(
        "--scene-downscale",
        type=int,
        default=1,
        help="Downscale factor for PySceneDetect. Use 1 to disable.",
    )
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def wait_for_file(path: str | Path, interval_sec: float = 1.0) -> None:
    path = Path(path)
    while not path.exists():
        print(f"Waiting for file: {path}", flush=True)
        time.sleep(interval_sec)


def replace_suffix(name: str, suffix: str | None) -> str:
    if suffix is None:
        return name
    if suffix == "":
        return str(Path(name).with_suffix(""))
    return str(Path(name).with_suffix(suffix))


def load_files_list(base_dir: str | Path, files_list_path: str | Path, with_extension: str | None, mkdir: bool = False) -> list[Path]:
    base_dir = Path(base_dir)
    files_list_path = Path(files_list_path)
    if files_list_path.exists():
        entries = [line.strip() for line in files_list_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
    else:
        entries = [
            str(path.relative_to(base_dir))
            for path in sorted(base_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    paths = []
    for entry in entries:
        entry = replace_suffix(entry, with_extension)
        path = Path(entry)
        if not path.is_absolute():
            path = base_dir / path
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        paths.append(path)
    return paths


def mirror_paths(reference_paths: list[Path], reference_base: str | Path, output_base: str | Path, with_extension: str, mkdir: bool = False) -> list[Path]:
    reference_base = Path(reference_base)
    output_base = Path(output_base)
    paths = []
    for ref_path in reference_paths:
        relative = Path(ref_path).relative_to(reference_base)
        out_path = output_base / \
            Path(replace_suffix(str(relative), with_extension))
        if mkdir:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        paths.append(out_path)
    return paths


def iter_subfolders(video_root: Path) -> list[str]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")
    return sorted(path.name for path in video_root.iterdir() if path.is_dir())


def probe_video(video_path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        return probe_video_pyav(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    streams = data.get("streams") or []
    return streams[0] if streams else {}


def probe_video_pyav(video_path: Path) -> dict:
    try:
        import av
    except ImportError:
        return {}
    try:
        with av.open(str(video_path)) as container:
            stream = next(s for s in container.streams if s.type == "video")
            return {
                "codec_name": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "nb_frames": str(stream.frames or 0),
                "avg_frame_rate": str(stream.average_rate) if stream.average_rate else None,
            }
    except Exception:
        return {}


def parse_frame_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_f = float(denominator)
        return float(numerator) / denominator_f if denominator_f else 0.0
    return float(value)


def codec_name(video_path: Path) -> str:
    return str(probe_video(video_path).get("codec_name") or "").lower()


def ffmpeg_has_decoder(decoder_name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and decoder_name in proc.stdout


def video_info_cv2(video_path: Path) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, frame_count


def video_info(video_path: Path) -> tuple[float, int]:
    try:
        return video_info_cv2(video_path)
    except Exception:
        stream = probe_video(video_path)
        fps = parse_frame_rate(stream.get("avg_frame_rate")
                               or stream.get("r_frame_rate"))
        nb_frames = stream.get("nb_frames")
        frame_count = int(nb_frames) if str(nb_frames or "").isdigit() else 0
        return fps, frame_count


def scene_ranges_from_video(video_path: Path, threshold: float, min_scene_len: int, backend: str | None = None, downscale: int = 1) -> np.ndarray:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise ImportError(
            "PySceneDetect is required. Install with: pip install scenedetect") from exc

    _, frame_count = video_info(video_path)
    video = open_video(
        str(video_path), backend=backend) if backend else open_video(str(video_path))
    scene_manager = SceneManager()
    if downscale > 1:
        if hasattr(scene_manager, "auto_downscale"):
            scene_manager.auto_downscale = False
        if hasattr(scene_manager, "downscale"):
            scene_manager.downscale = int(downscale)
    scene_manager.add_detector(ContentDetector(
        threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, show_progress=True)
    scenes = scene_manager.get_scene_list()

    ranges = []
    for start_time, end_time in scenes:
        start_frame = max(0, int(start_time.get_frames()))
        end_frame = max(start_frame, int(end_time.get_frames()) - 1)
        if frame_count > 0:
            end_frame = min(end_frame, frame_count - 1)
        ranges.append((start_frame, end_frame))

    if not ranges and frame_count > 0:
        ranges.append((0, frame_count - 1))
    return np.asarray(ranges, dtype=np.int64)


def transcode_to_h264(src_path: Path, dst_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg executable not found. Install ffmpeg for AV1 fallback transcoding.")
    input_args = []
    if codec_name(src_path) == "av1" and ffmpeg_has_decoder("av1_cuvid"):
        input_args = ["-hwaccel", "cuda", "-c:v", "av1_cuvid"]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *input_args,
        "-i",
        str(src_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",
        str(dst_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg fallback transcode failed for {src_path}: {proc.stderr[-1000:]}")


def detect_scene_ranges(video_path: Path, threshold: float, min_scene_len: int, downscale: int = 1) -> np.ndarray:
    codec = codec_name(video_path)
    if codec == "av1":
        with tempfile.TemporaryDirectory(prefix="aic_av1_scene_") as tmp_dir:
            tmp_path = Path(tmp_dir) / f"{video_path.stem}_h264.mp4"
            print(
                "Running PySceneDetect after temporary FFmpeg/NVDEC H.264 transcode.", flush=True)
            transcode_to_h264(video_path, tmp_path)
            return scene_ranges_from_video(tmp_path, threshold, min_scene_len, downscale=downscale)

    try:
        return scene_ranges_from_video(video_path, threshold, min_scene_len, downscale=downscale)
    except Exception as exc:
        print(
            f"Normal scene detection failed for {video_path} (codec={codec or 'unknown'}): {exc}", flush=True)

    try:
        print("Retrying scene detection with PySceneDetect pyav backend.", flush=True)
        return scene_ranges_from_video(video_path, threshold, min_scene_len, backend="pyav", downscale=downscale)
    except Exception as pyav_exc:
        print(
            f"PyAV scene detection fallback failed for {video_path}: {pyav_exc}", flush=True)

    with tempfile.TemporaryDirectory(prefix="aic_av1_scene_") as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{video_path.stem}_h264.mp4"
        print(
            "Retrying scene detection after temporary FFmpeg H.264 transcode.", flush=True)
        transcode_to_h264(video_path, tmp_path)
        return scene_ranges_from_video(tmp_path, threshold, min_scene_len, downscale=downscale)


def main() -> None:
    args = parse_args()
    video_root = PROJECT_ROOT / "dataraw" / "videos" / "Video"
    files_list_root = PROJECT_ROOT / "dataraw" / "folder_file_list"
    scene_root = PROJECT_ROOT / "ProcessedData" / "scence_boundary"

    for subfolder in iter_subfolders(video_root):
        subfolder_path = video_root / subfolder
        print(f"\nScene detection (pyscenedetect_content): {subfolder}")
        videos_path = load_files_list(
            base_dir=subfolder_path,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=None,
        )
        outputs_path = mirror_paths(
            videos_path,
            subfolder_path,
            scene_root / subfolder,
            ".txt",
            mkdir=True,
        )
        if args.limit_videos > 0:
            videos_path = videos_path[: args.limit_videos]
            outputs_path = outputs_path[: args.limit_videos]

        for video_path, output_path in tqdm(list(zip(videos_path, outputs_path)), desc=subfolder):
            if output_path.exists() and not args.overwrite:
                continue
            wait_for_file(video_path)
            scene_ranges = detect_scene_ranges(
                video_path, args.content_threshold, args.min_scene_len, args.scene_downscale)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_path, scene_ranges, fmt="%d %d")
            print(f"{video_path.name}: {len(scene_ranges)} scenes -> {output_path}")


if __name__ == "__main__":
    main()
