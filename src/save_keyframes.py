from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import time
from multiprocessing import JoinableQueue
from pathlib import Path

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / \
    "video_processing" if SCRIPT_DIR.name == "final" else SCRIPT_DIR
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select keyframe indices from old scene/embedding outputs, then save keyframe webp files and maps."
    )
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--clean-frames-embedding", action="store_true")
    parser.add_argument("--skip-selection", action="store_true",
                        help="Use existing keyframes_indices_B32_* files only.")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_num_threads(cfg_threads: int) -> int:
    if cfg_threads == -1:
        cpu = os.cpu_count() or 4
        return max(1, cpu - 4)
    return cfg_threads


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


def mirror_dirs(reference_paths: list[Path], reference_base: str | Path, output_base: str | Path, mkdir: bool = False) -> list[Path]:
    reference_base = Path(reference_base)
    output_base = Path(output_base)
    paths = []
    for ref_path in reference_paths:
        relative = Path(ref_path).relative_to(reference_base).with_suffix("")
        out_path = output_base / relative
        if mkdir:
            out_path.mkdir(parents=True, exist_ok=True)
        paths.append(out_path)
    return paths


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
        "stream=codec_name,nb_frames,avg_frame_rate,r_frame_rate",
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


def video_fps(video_path: Path) -> float:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if fps > 0:
                return fps
        cap.release()
    except Exception:
        pass
    stream = probe_video(video_path)
    return parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))


def hierarchical_clustering_avg(features: np.ndarray) -> list[int]:
    try:
        import scipy.spatial.distance
        from sklearn.metrics import silhouette_score
    except ImportError:
        center = features.mean(axis=0)
        return [int(np.argmin(((features - center[None, :]) ** 2).sum(axis=1)))]

    n = len(features)
    labels = np.arange(n)
    best_score = -float("inf")
    best_labels = labels.copy()
    best_num_clusters = n
    full_dist = scipy.spatial.distance.cdist(features, features)

    def cluster_dist(id1: int, id2: int) -> float:
        mask1 = labels == id1
        mask2 = labels == id2
        return float(full_dist[np.ix_(mask1, mask2)].mean())

    consec_dists = np.zeros(n - 1)
    for i in range(n - 1):
        consec_dists[i] = cluster_dist(i, i + 1)

    maximum_num_clusters = int(n**0.5) + 1
    num_clusters = n
    while num_clusters >= 3 and len(consec_dists) > 0:
        min_id = int(np.argmin(consec_dists))
        min_id_first, min_id_sec = min_id, min_id + 1
        labels = np.where(labels >= min_id_sec, labels - 1, labels)
        num_clusters -= 1
        consec_dists[min_id_sec:-1] = consec_dists[min_id_sec + 1:]
        consec_dists = consec_dists[:-1]
        if min_id_first < len(consec_dists):
            consec_dists[min_id_first] = cluster_dist(
                min_id_first, min_id_first + 1)
        if min_id_first >= 1:
            consec_dists[min_id_first -
                         1] = cluster_dist(min_id_first - 1, min_id_first)
        if num_clusters <= maximum_num_clusters:
            try:
                score = silhouette_score(features, labels)
            except ValueError:
                continue
            if score > best_score:
                best_score = float(score)
                best_labels = labels.copy()
                best_num_clusters = num_clusters

    selected = []
    for cluster_id in range(best_num_clusters):
        cluster_mask = best_labels == cluster_id
        if not np.any(cluster_mask):
            continue
        cluster = features[cluster_mask]
        center = cluster.mean(axis=0)
        local_idx = int(
            np.argmin(((cluster - center[None, :]) ** 2).sum(axis=1)))
        selected.append(int(np.flatnonzero(cluster_mask)[local_idx]))
    return sorted(selected)


def process_scene(scene_boundary: np.ndarray, frame_embs: np.ndarray) -> list[int]:
    left, right = [int(x) for x in scene_boundary[:2]]
    left = max(0, left)
    right = min(len(frame_embs) - 1, right)
    if right < left:
        return []
    scene_embs = frame_embs[left: right + 1]
    if len(scene_embs) < 5:
        center = scene_embs.mean(axis=0)
        return [left + int(np.argmin(((scene_embs - center[None, :]) ** 2).sum(axis=1)))]
    return [left + idx for idx in hierarchical_clustering_avg(scene_embs)]


def select_keyframes(scenes_boundary_path: Path, frames_embedding_path: Path, output_path: Path) -> list[int]:
    scenes_boundary = np.loadtxt(scenes_boundary_path, dtype=np.int64)
    if scenes_boundary.size == 0:
        scenes_boundary = np.empty((0, 2), dtype=np.int64)
    elif scenes_boundary.ndim == 1:
        scenes_boundary = scenes_boundary.reshape(1, -1)
    frame_embs = np.load(frames_embedding_path)
    keyframe_indices: list[int] = []
    for scene_boundary in tqdm(scenes_boundary, desc=frames_embedding_path.stem, leave=False):
        keyframe_indices.extend(process_scene(scene_boundary, frame_embs))
    keyframe_indices = sorted(set(keyframe_indices))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.asarray(
        keyframe_indices, dtype=np.int64), fmt="%d")
    return keyframe_indices


def load_keyframe_indices(keyframe_indices_path: Path) -> np.ndarray:
    keyframe_indices = np.loadtxt(keyframe_indices_path, dtype=np.int64)
    if keyframe_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.atleast_1d(keyframe_indices).astype(np.int64))


def write_mapping(output_path_map_keyframes: Path, rows: list[tuple]) -> None:
    output_path_map_keyframes.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_map_keyframes, "w", encoding="utf-8", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(("id", "pts_time", "fps", "frame_idx"))
        csvwriter.writerows(rows)


def extract_frame_ffmpeg(video_path: Path, frame_idx: int, output_file: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg executable not found. Install ffmpeg for AV1 frame extraction fallback.")
    # select uses zero-based decoded frame numbers, matching OpenCV frame_idx.
    vf = f"select=eq(n\\,{int(frame_idx)})"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(output_file),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0 or not output_file.exists():
        raise RuntimeError(
            f"FFmpeg could not extract frame {frame_idx} from {video_path}: {proc.stderr[-1000:]}")


def extract_frame_pyav(video_path: Path, frame_idx: int, output_file: Path) -> None:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is not installed. Install av for AV1 frame extraction fallback.") from exc

    with av.open(str(video_path)) as container:
        video_stream = next(s for s in container.streams if s.type == "video")
        for decoded_idx, frame in enumerate(container.decode(video_stream)):
            if decoded_idx == int(frame_idx):
                output_file.parent.mkdir(parents=True, exist_ok=True)
                frame.to_image().convert("RGB").save(output_file, format="WEBP", quality=80)
                return
    raise RuntimeError(
        f"PyAV could not find frame {frame_idx} in {video_path}")


def extract_frame_fallback(video_path: Path, frame_idx: int, output_file: Path) -> None:
    try:
        extract_frame_pyav(video_path, frame_idx, output_file)
    except Exception as pyav_exc:
        print(
            f"PyAV frame extraction fallback failed for {video_path} frame {frame_idx}: {pyav_exc}", flush=True)
        extract_frame_ffmpeg(video_path, frame_idx, output_file)


def save_frames_cv2(keyframe_indices: np.ndarray, video_path: Path, output_path: Path, webp_quality: int) -> list[tuple]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames_mapping = []
    for key_frame_idx, frame_idx in enumerate(keyframe_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            file_path = output_path / f"{key_frame_idx:06d}.webp"
            extract_frame_fallback(video_path, int(frame_idx), file_path)
            frames_mapping.append(
                (key_frame_idx, frame_idx / fps if fps > 0 else "", fps, int(frame_idx)))
            continue
        file_path = output_path / f"{key_frame_idx:06d}.webp"
        if cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_WEBP_QUALITY, int(webp_quality)]):
            frames_mapping.append(
                (key_frame_idx, frame_idx / fps if fps > 0 else "", fps, int(frame_idx)))
    cap.release()
    return frames_mapping


def save_frames_ffmpeg(keyframe_indices: np.ndarray, video_path: Path, output_path: Path) -> list[tuple]:
    fps = video_fps(video_path)
    rows = []
    for key_frame_idx, frame_idx in enumerate(keyframe_indices):
        file_path = output_path / f"{key_frame_idx:06d}.webp"
        extract_frame_fallback(video_path, int(frame_idx), file_path)
        rows.append((key_frame_idx, frame_idx / fps if fps >
                    0 else "", fps, int(frame_idx)))
    return rows


def save_frames(keyframe_indices_path: Path, video_path: Path, output_path: Path, output_path_map_keyframes: Path, webp_quality: int) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    keyframe_indices = load_keyframe_indices(keyframe_indices_path)
    try:
        rows = save_frames_cv2(
            keyframe_indices, video_path, output_path, webp_quality)
    except Exception as exc:
        codec = codec_name(video_path)
        print(
            f"Normal keyframe saving failed for {video_path} (codec={codec or 'unknown'}): {exc}", flush=True)
        print("Retrying keyframe extraction with FFmpeg fallback for AV1/unsupported codecs.", flush=True)
        rows = save_frames_ffmpeg(keyframe_indices, video_path, output_path)
    write_mapping(output_path_map_keyframes, rows)


def process_one(q: JoinableQueue):
    while True:
        try:
            item = q.get(block=False)
        except queue.Empty:
            break
        try:
            if len(item) == 5:
                keyframes_indices_path, video_path, output_path, output_path_map_keyframes, webp_quality = item
            else:
                keyframes_indices_path, video_path, output_path, output_path_map_keyframes = item
                webp_quality = 80
            save_frames(
                Path(keyframes_indices_path),
                Path(video_path),
                Path(output_path),
                Path(output_path_map_keyframes),
                int(webp_quality),
            )
        except Exception as e:
            print(f"{item[1]}: {e}", flush=True)
        finally:
            q.task_done()


def main() -> None:
    args = parse_args()
    emb_root = PROJECT_ROOT / "dataraw" / "embeddings"
    video_root = PROJECT_ROOT / "dataraw" / "videos" / "Video"
    files_list_root = PROJECT_ROOT / "dataraw" / "folder_file_list"
    scene_root = PROJECT_ROOT / "ProcessedData" / "scence_boundary"
    keyframe_root = PROJECT_ROOT / "ProcessedData" / "data" / "keyframes"
    map_root = PROJECT_ROOT / "ProcessedData" / "data" / "map_keyframes"

    for subfolder in iter_subfolders(video_root):
        print(f"\nKeyframe selection/save: {subfolder}")
        videos_path = load_files_list(
            base_dir=video_root / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=None,
        )
        subfolder_base = video_root / subfolder
        embs_path = mirror_paths(
            videos_path,
            subfolder_base,
            emb_root / subfolder,
            ".npy",
        )
        scenes_bound_path = mirror_paths(
            videos_path,
            subfolder_base,
            scene_root / subfolder,
            ".txt",
        )
        keyframes_ids_path = mirror_paths(
            videos_path,
            subfolder_base,
            emb_root / f"keyframes_indices_B32_{subfolder}",
            ".txt",
            mkdir=True,
        )
        out_keyframes_path = mirror_dirs(
            videos_path,
            subfolder_base,
            keyframe_root / subfolder,
            mkdir=True,
        )
        out_mappings_path = mirror_paths(
            videos_path,
            subfolder_base,
            map_root / subfolder,
            ".csv",
            mkdir=True,
        )
        if args.limit_videos > 0:
            videos_path = videos_path[: args.limit_videos]
            embs_path = embs_path[: args.limit_videos]
            scenes_bound_path = scenes_bound_path[: args.limit_videos]
            keyframes_ids_path = keyframes_ids_path[: args.limit_videos]
            out_keyframes_path = out_keyframes_path[: args.limit_videos]
            out_mappings_path = out_mappings_path[: args.limit_videos]

        for video_path, emb_path, scene_path, kf_path, out_kf_path, out_map_path in tqdm(
            list(zip(videos_path, embs_path, scenes_bound_path,
                 keyframes_ids_path, out_keyframes_path, out_mappings_path)),
            desc=subfolder,
        ):
            wait_for_file(video_path)
            if not args.skip_selection:
                wait_for_file(emb_path)
                wait_for_file(scene_path)
                if args.overwrite or not kf_path.exists():
                    select_keyframes(scene_path, emb_path, kf_path)
                if args.clean_frames_embedding:
                    open(emb_path, "w").close()
            wait_for_file(kf_path)
            if out_map_path.exists() and not args.overwrite:
                continue
            save_frames(kf_path, video_path, out_kf_path,
                        out_map_path, args.webp_quality)


if __name__ == "__main__":
    main()
