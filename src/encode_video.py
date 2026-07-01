from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent / \
    "video_processing" if SCRIPT_DIR.name == "final" else SCRIPT_DIR

MODEL_ARCH = "ViT-H-14"
MODEL_PRETRAINED = "laion2b_s32b_b79k"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode full video frames using OpenCLIP ViT-H-14, keeping the old dataraw/embeddings flow."
    )
    parser.add_argument("--model-arch", default=MODEL_ARCH)
    parser.add_argument("--model-pretrained", default=MODEL_PRETRAINED)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-every", type=int, default=1)
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


def load_model(args: argparse.Namespace):
    import open_clip

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_arch,
        pretrained=args.model_pretrained or None,
        jit=args.jit,
        device=device,
    )
    model.eval()
    if device.type == "cuda":
        model = model.half()
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess, device


def load_video(vid_path, new_width, new_height):
    try:
        import cv2

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {vid_path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(new_width), int(
                new_height)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        if frames:
            return np.stack(frames, axis=0)
        raise RuntimeError(f"No frames decoded from {vid_path}")
    except Exception as exc:
        print(f"Normal video load failed for {vid_path}: {exc}", flush=True)
        return load_video_pyav(Path(vid_path), int(new_width), int(new_height))


def load_video_pyav(video_path: Path, new_width: int, new_height: int) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required for AV1 fallback decoding without ffmpeg.exe.") from exc

    frames = []
    with av.open(str(video_path)) as container:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            image = frame.to_image().convert("RGB").resize((new_width, new_height))
            frames.append(np.asarray(image, dtype=np.uint8))
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def get_resized_imgsize(width, height, model_input_size):
    if width > height:
        new_width = min(int(width), int(model_input_size))
        new_height = int((new_width / width) * height)
    else:
        new_height = min(int(height), int(model_input_size))
        new_width = int((new_height / height) * width)
    return max(1, new_width), max(1, new_height)


def get_new_video_size(video_path, model_input_size=224):
    stream = probe_video(Path(video_path))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video size for {video_path}")
    return get_resized_imgsize(width, height, model_input_size)


class FrameLoader(Dataset):
    def __init__(self, vid_frames, img_preprocess):
        super().__init__()
        self.vid_frames = vid_frames
        self.img_preprocess = img_preprocess

    def __len__(self):
        return len(self.vid_frames)

    def __getitem__(self, idx):
        frame = self.vid_frames[idx]
        image = Image.fromarray(frame)
        return self.img_preprocess(image)


def load_video_producer(vid_paths, model_input_size, q):
    for vpath in vid_paths:
        while q.qsize() >= 2:
            time.sleep(1)
        wait_for_file(vpath)
        new_width, new_height = get_new_video_size(vpath, model_input_size)
        video = load_video(vpath, new_width, new_height)
        q.put(video)
    q.join()


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


def encode_tensor_batch(model, batch: list[torch.Tensor], device: torch.device) -> np.ndarray:
    tensor = torch.stack(batch).to(device, non_blocking=True)
    if device.type == "cuda":
        tensor = tensor.half()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.float().cpu().numpy().astype(np.float16)


def preprocess_normalize_values(preprocess) -> tuple[tuple[float, ...], tuple[float, ...]]:
    for transform in getattr(preprocess, "transforms", []):
        mean = getattr(transform, "mean", None)
        std = getattr(transform, "std", None)
        if mean is not None and std is not None:
            return tuple(float(x) for x in mean), tuple(float(x) for x in std)
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def preprocess_image_size(preprocess, default: int = 224) -> int:
    for transform in getattr(preprocess, "transforms", []):
        size = getattr(transform, "size", None)
        if isinstance(size, int):
            return size
        if isinstance(size, (tuple, list)) and size:
            return int(size[0])
    return default


def encode_numpy_batch(model, frames: list[np.ndarray], device: torch.device, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(frames)).permute(
        0, 3, 1, 2).contiguous()
    tensor = tensor.to(device, non_blocking=True).float().div_(255.0)
    tensor = tensor.sub_(mean).div_(std)
    if device.type == "cuda":
        tensor = tensor.half()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.float().cpu().numpy().astype(np.float16)


def save_embedding_chunks(output_path: Path, chunks: list[np.ndarray]) -> None:
    if not chunks:
        raise RuntimeError(f"No frames encoded for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.vstack(chunks).astype(np.float16))
    print(f"Saved: {output_path}")


def encode_video_cv2(video_path: Path, output_path: Path, model, preprocess, device: torch.device, batch_size: int, sample_every: int) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = -1
    batch: list[torch.Tensor] = []
    chunks: list[np.ndarray] = []

    pbar = tqdm(total=frame_count if frame_count > 0 else None,
                desc=f"Encoding {video_path.name}", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        pbar.update(1)
        if sample_every > 1 and frame_idx % sample_every != 0:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        batch.append(preprocess(image))
        if len(batch) >= batch_size:
            chunks.append(encode_tensor_batch(model, batch, device))
            batch = []

    pbar.close()
    cap.release()
    if batch:
        chunks.append(encode_tensor_batch(model, batch, device))
    if not chunks:
        raise RuntimeError(f"No frames encoded from {video_path}")
    save_embedding_chunks(output_path, chunks)


def encode_video_ffmpeg(video_path: Path, output_path: Path, model, preprocess, device: torch.device, batch_size: int, sample_every: int, use_nvdec: bool = True) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg executable not found. Install ffmpeg for AV1 fallback decoding.")

    stream = probe_video(video_path)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    frame_count = int(stream.get("nb_frames") or 0) if str(
        stream.get("nb_frames") or "").isdigit() else 0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not probe width/height for {video_path}")

    image_size = preprocess_image_size(preprocess)
    output_width = output_height = image_size
    mean_values, std_values = preprocess_normalize_values(preprocess)
    mean = torch.tensor(mean_values, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=device).view(1, 3, 1, 1)

    input_args = []
    output_args = [
        "-vf",
        (
            f"scale='if(gt(a,1),-2,{image_size})':"
            f"'if(gt(a,1),{image_size},-2)':flags=bicubic,"
            f"crop={image_size}:{image_size}"
        ),
    ]
    codec = str(stream.get("codec_name") or "").lower()
    if use_nvdec and codec == "av1" and ffmpeg_has_decoder("av1_cuvid"):
        input_args = [
            "-hwaccel",
            "cuda",
            "-c:v",
            "av1_cuvid",
        ]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-i",
        str(video_path),
        *output_args,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_size = output_width * output_height * 3
    frame_idx = -1
    batch: list[np.ndarray] = []
    chunks: list[np.ndarray] = []

    decode_label = "FFmpeg NVDEC" if input_args else "FFmpeg decode"
    pbar = tqdm(total=frame_count if frame_count > 0 else None,
                desc=f"{decode_label} {video_path.name}", unit="frame")
    while True:
        raw = proc.stdout.read(frame_size)
        if not raw:
            break
        if len(raw) != frame_size:
            proc.kill()
            raise RuntimeError(
                f"Incomplete raw frame while decoding {video_path}")
        frame_idx += 1
        pbar.update(1)
        if sample_every > 1 and frame_idx % sample_every != 0:
            continue
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (output_height, output_width, 3))
        batch.append(frame.copy())
        if len(batch) >= batch_size:
            chunks.append(encode_numpy_batch(model, batch, device, mean, std))
            batch = []

    pbar.close()
    stderr = proc.stderr.read().decode(
        "utf-8", errors="replace") if proc.stderr is not None else ""
    return_code = proc.wait()
    if return_code != 0 and not chunks and not batch:
        raise RuntimeError(
            f"FFmpeg decode failed for {video_path}: {stderr[-1000:]}")
    if batch:
        chunks.append(encode_numpy_batch(model, batch, device, mean, std))
    save_embedding_chunks(output_path, chunks)


def encode_video_pyav(video_path: Path, output_path: Path, model, preprocess, device: torch.device, batch_size: int, sample_every: int) -> None:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is not installed. Install av for AV1 fallback decoding without ffmpeg.exe.") from exc

    stream_info = probe_video_pyav(video_path)
    frame_count = int(stream_info.get("nb_frames") or 0) if str(
        stream_info.get("nb_frames") or "").isdigit() else 0
    batch: list[torch.Tensor] = []
    chunks: list[np.ndarray] = []
    decoded_idx = -1
    with av.open(str(video_path)) as container:
        video_stream = next(s for s in container.streams if s.type == "video")
        pbar = tqdm(total=frame_count if frame_count > 0 else None,
                    desc=f"PyAV decode {video_path.name}", unit="frame")
        for frame in container.decode(video_stream):
            decoded_idx += 1
            pbar.update(1)
            if sample_every > 1 and decoded_idx % sample_every != 0:
                continue
            batch.append(preprocess(frame.to_image().convert("RGB")))
            if len(batch) >= batch_size:
                chunks.append(encode_tensor_batch(model, batch, device))
                batch = []
        pbar.close()
    if batch:
        chunks.append(encode_tensor_batch(model, batch, device))
    save_embedding_chunks(output_path, chunks)


def encode_video(video_path: Path, output_path: Path, model, preprocess, device: torch.device, batch_size: int, sample_every: int) -> None:
    codec = codec_name(video_path)
    if codec == "av1":
        try:
            print("Using FFmpeg/NVDEC for AV1 decode.", flush=True)
            encode_video_ffmpeg(video_path, output_path, model,
                                preprocess, device, batch_size, sample_every)
            return
        except Exception as ffmpeg_exc:
            print(
                f"FFmpeg/NVDEC decode failed for {video_path}: {ffmpeg_exc}", flush=True)
            print("Retrying with PyAV CPU fallback for AV1.", flush=True)
            encode_video_pyav(video_path, output_path, model,
                              preprocess, device, batch_size, sample_every)
            return

    try:
        encode_video_cv2(video_path, output_path, model,
                         preprocess, device, batch_size, sample_every)
    except Exception as exc:
        print(
            f"Normal decode failed for {video_path} (codec={codec or 'unknown'}): {exc}", flush=True)
        try:
            print("Retrying with FFmpeg/NVDEC fallback for AV1/unsupported codecs.", flush=True)
            encode_video_ffmpeg(video_path, output_path, model,
                                preprocess, device, batch_size, sample_every)
            return
        except Exception as ffmpeg_exc:
            print(
                f"FFmpeg/NVDEC decode fallback failed for {video_path}: {ffmpeg_exc}", flush=True)
        print("Retrying with PyAV CPU fallback for AV1/unsupported codecs.", flush=True)
        encode_video_pyav(video_path, output_path, model,
                          preprocess, device, batch_size, sample_every)


def process_video(
    video_path,
    output_path,
    video_frames,
    img_preprocess,
    model,
    batch_size,
    num_workers,
):
    device = next(model.parameters()).device
    if video_frames is None:
        encode_video(Path(video_path), Path(output_path), model,
                     img_preprocess, device, batch_size, sample_every=1)
        return

    fdl = DataLoader(
        FrameLoader(video_frames, img_preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    frames_emb = []
    for batch in tqdm(fdl, desc=f"Encoding {os.path.basename(video_path)}"):
        batch = batch.to(device, non_blocking=True)
        if device.type == "cuda":
            batch = batch.half()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            image_features = model.encode_image(batch)
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)
        frames_emb.append(image_features.float().cpu().numpy())

    if not frames_emb:
        raise RuntimeError(f"No frames encoded from {video_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.vstack(frames_emb).astype(np.float16))
    print(f"Saved: {output_path}")


def skip_completted(videos_path, outputs_path):
    new_videos_path = []
    new_outputs_path = []
    for vp, op in zip(videos_path, outputs_path):
        if os.path.exists(op):
            continue
        new_videos_path.append(vp)
        new_outputs_path.append(op)
    return new_videos_path, new_outputs_path


def main() -> None:
    args = parse_args()
    video_root = PROJECT_ROOT / "dataraw" / "videos" / "Video"
    files_list_root = PROJECT_ROOT / "dataraw" / "folder_file_list"
    emb_root = PROJECT_ROOT / "dataraw" / "embeddings"

    model, preprocess, device = load_model(args)
    for subfolder in iter_subfolders(video_root):
        print(f"\nProcessing subfolder: {subfolder}")
        videos_path = load_files_list(
            base_dir=video_root / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=None,
        )
        outputs_path = mirror_paths(
            videos_path,
            video_root / subfolder,
            emb_root / subfolder,
            ".npy",
            mkdir=True,
        )
        if args.limit_videos > 0:
            videos_path = videos_path[: args.limit_videos]
            outputs_path = outputs_path[: args.limit_videos]
        pending = list(zip(videos_path, outputs_path)) if args.overwrite else list(
            zip(*skip_completted(videos_path, outputs_path)))
        if not pending:
            print(f"All videos already processed, skipping {subfolder}")
            continue
        for video_path, output_path in tqdm(pending, desc=subfolder):
            wait_for_file(video_path)
            encode_video(video_path, output_path, model, preprocess,
                         device, args.batch_size, args.sample_every)


if __name__ == "__main__":
    main()
