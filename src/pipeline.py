from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from encode_video import (
    MODEL_ARCH,
    MODEL_PRETRAINED,
    encode_video,
    load_model,
    wait_for_file,
)
from save_keyframes import save_frames, select_keyframes
from scene_boundary import detect_scene_ranges


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full mp4 pipeline: encode_video -> scene_boundary -> "
            "clustering/select keyframes -> save_keyframes."
        )
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=PROJECT_ROOT / "dataraw" / "videos" / "Video",
        help="Root folder containing .mp4 files. Files are scanned recursively.",
    )
    parser.add_argument("--model-arch", default=MODEL_ARCH)
    parser.add_argument("--model-pretrained", default=MODEL_PRETRAINED)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--content-threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=int, default=15)
    parser.add_argument(
        "--scene-downscale",
        type=int,
        default=2,
        help="Downscale factor for PySceneDetect. Use 1 to disable.",
    )
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the sorted video list into this many shards for parallel runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Run only this zero-based shard index when --num-shards > 1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--skip-scene-boundary", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-save-keyframes", action="store_true")
    parser.add_argument("--clean-frames-embedding", action="store_true")
    return parser.parse_args()


def find_mp4_files(video_root: Path) -> list[Path]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")
    return sorted(path for path in video_root.rglob("*.mp4") if path.is_file())


def require_file(path: Path, producer_step: str, skip_flag: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"Required file not found: {path}\n"
        f"Run the {producer_step} step first, or remove {skip_flag}."
    )


def split_collection(relative_video_path: Path) -> tuple[str, Path]:
    parts = relative_video_path.parts
    if len(parts) <= 1:
        return "root", relative_video_path
    return parts[0], Path(*parts[1:])


def output_paths(video_path: Path, video_root: Path) -> dict[str, Path]:
    relative_path = video_path.relative_to(video_root)
    collection, relative_in_collection = split_collection(relative_path)

    emb_root = PROJECT_ROOT / "dataraw" / "embeddings"
    scene_root = PROJECT_ROOT / "ProcessedData" / "scence_boundary"
    keyframe_root = PROJECT_ROOT / "ProcessedData" / "data" / "keyframes"
    map_root = PROJECT_ROOT / "ProcessedData" / "data" / "map_keyframes"

    return {
        "embedding": emb_root / relative_path.with_suffix(".npy"),
        "scene": scene_root / relative_path.with_suffix(".txt"),
        "keyframe_indices": emb_root
        / f"keyframes_indices_B32_{collection}"
        / relative_in_collection.with_suffix(".txt"),
        "keyframes": keyframe_root / relative_path.with_suffix(""),
        "map": map_root / relative_path.with_suffix(".csv"),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    video_root = args.video_root.resolve()
    videos = find_mp4_files(video_root)
    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]
    if args.num_shards > 1:
        videos = videos[args.shard_index:: args.num_shards]
    if not videos:
        print(f"No .mp4 files found in {video_root}")
        return

    model = preprocess = device = None
    if not args.skip_encode:
        model, preprocess, device = load_model(args)

    for video_path in tqdm(videos, desc="Full pipeline"):
        paths = output_paths(video_path, video_root)
        wait_for_file(video_path)

        if not args.skip_encode:
            if args.overwrite or not paths["embedding"].exists():
                paths["embedding"].parent.mkdir(parents=True, exist_ok=True)
                encode_video(
                    video_path,
                    paths["embedding"],
                    model,
                    preprocess,
                    device,
                    args.batch_size,
                    sample_every=1,
                )

        if not args.skip_scene_boundary:
            if args.overwrite or not paths["scene"].exists():
                scene_ranges = detect_scene_ranges(
                    video_path,
                    args.content_threshold,
                    args.min_scene_len,
                    args.scene_downscale,
                )
                paths["scene"].parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(paths["scene"], scene_ranges, fmt="%d %d")
                print(f"Scene boundary: {video_path.name} -> {paths['scene']}")

        if not args.skip_clustering:
            require_file(paths["embedding"], "encode_video", "--skip-encode")
            require_file(paths["scene"], "scene_boundary", "--skip-scene-boundary")
            if args.overwrite or not paths["keyframe_indices"].exists():
                select_keyframes(
                    paths["scene"],
                    paths["embedding"],
                    paths["keyframe_indices"],
                )
                print(f"Keyframe indices: {video_path.name} -> {paths['keyframe_indices']}")
            if args.clean_frames_embedding:
                paths["embedding"].write_bytes(b"")

        if not args.skip_save_keyframes:
            require_file(paths["keyframe_indices"], "clustering", "--skip-clustering")
            if args.overwrite or not paths["map"].exists():
                save_frames(
                    paths["keyframe_indices"],
                    video_path,
                    paths["keyframes"],
                    paths["map"],
                    args.webp_quality,
                )
                print(f"Saved keyframes: {video_path.name} -> {paths['keyframes']}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
