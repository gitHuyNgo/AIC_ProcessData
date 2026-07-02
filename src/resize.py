import sys
from pathlib import Path
import os
import cv2
import subprocess
from multiprocessing import JoinableQueue, Process
import queue
import hydra
from omegaconf import DictConfig
from typing import List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "config"

from utils import load_files_list, wait_for_file


NVENC_PARALLEL_STREAMS = 4


def get_num_threads(cfg_threads: int) -> int:
    if cfg_threads == -1:
        cpu = os.cpu_count() or 4
        return max(1, cpu - 2)
    return cfg_threads


def resize_keyframes_batch(batch: List[Tuple[str, str]], height: int):
    created_dirs = set()
    for src_path, dst_path in batch:
        dst_dir = os.path.dirname(dst_path)
        if dst_dir not in created_dirs:
            os.makedirs(dst_dir, exist_ok=True)
            created_dirs.add(dst_dir)

        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        new_w = int(w * (height / h))

        resized = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(dst_path, resized, [cv2.IMWRITE_WEBP_QUALITY, 80])


def resize_keyframes_worker(q: JoinableQueue, height: int, batch_size: int = 128):
    batch = []

    while True:
        try:
            item = q.get(timeout=1)
            if item is None:
                try:
                    if batch:
                        resize_keyframes_batch(batch, height)
                except Exception as e:
                    print(f"Error resizing keyframes batch: {e}")
                finally:
                    q.task_done()
                break
            
            try:
                src_path, dst_path = item
                batch.append((src_path, dst_path))

                if len(batch) >= batch_size:
                    resize_keyframes_batch(batch, height)
                    batch = []
            finally:
                q.task_done()

        except queue.Empty:
            continue


_AV1_HW_DECODE_SUPPORTED: Optional[bool] = None


def check_av1_hw_decode() -> bool:
    global _AV1_HW_DECODE_SUPPORTED
    if _AV1_HW_DECODE_SUPPORTED is None:
        try:
            cmd = ["ffmpeg", "-hide_banner", "-decoders"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            _AV1_HW_DECODE_SUPPORTED = (
                "av1_cuvid" in result.stdout
                or
                "av1_nvdec" in result.stdout
            )
        except Exception:
            _AV1_HW_DECODE_SUPPORTED = False
    return _AV1_HW_DECODE_SUPPORTED


def get_video_codec(src_video: str) -> str:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        src_video
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().lower()
    except subprocess.CalledProcessError:
        return ""


def _run_ffmpeg(src_video: str, dst_video: str, height: int, bitrate: str, hw_decode: bool) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if hw_decode:
        cmd.extend([
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", src_video,
            "-vf", f"scale_cuda=-2:{height}"
        ])
    else:
        cmd.extend([
            "-i", src_video,
            "-vf", f"hwupload_cuda,scale_cuda=-2:{height}"
        ])

    cmd.extend([
        "-c:v", "hevc_nvenc",
        "-preset", "p3",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "22",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", f"{int(bitrate.replace('M', '')) * 2}M",
        "-spatial_aq", "1",
        "-temporal_aq", "1",
        "-rc-lookahead", "32",
        "-surfaces", "64",
        "-bf", "3",
        "-b_ref_mode", "middle",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-y",
        dst_video,
    ])

    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except Exception:
        return False


def encode_video_fast(src_video: str, dst_video: str, height: int, bitrate: str):
    os.makedirs(os.path.dirname(dst_video), exist_ok=True)

    if os.path.exists(dst_video):
        return

    codec = get_video_codec(src_video)
    use_gpu_decode = True

    if codec == "h264":
        print("[GPU] H264 -> GPU decode")
    elif codec == "hevc":
        print("[GPU] HEVC -> GPU decode")
    elif codec == "av1":
        if check_av1_hw_decode():
            print("[GPU] AV1 -> GPU decode")
        else:
            print("[CPU] AV1 -> CPU decode fallback")
            use_gpu_decode = False
    else:
        print(f"[CPU] {codec.upper() if codec else 'UNKNOWN'} -> CPU decode fallback")
        use_gpu_decode = False

    success = False
    if use_gpu_decode:
        success = _run_ffmpeg(src_video, dst_video, height, bitrate, hw_decode=True)
        if not success:
            print("[Retry] GPU decode failed, retrying with CPU decode...")

    if not success:
        _run_ffmpeg(src_video, dst_video, height, bitrate, hw_decode=False)


def encode_video_worker(q: JoinableQueue, height: int, bitrate: str):
    while True:
        try:
            item = q.get(timeout=1)
            if item is None:
                q.task_done()
                break
            
            try:
                src_video, dst_video = item
                encode_video_fast(src_video, dst_video, height, bitrate)
            finally:
                q.task_done()

        except queue.Empty:
            continue


# =========================
# Main orchestrator
# =========================


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(config: DictConfig):
    keyframe_root = PROJECT_ROOT / "ProcessedData" / "data" / "keyframes"
    video_root = PROJECT_ROOT / "dataraw" / "videos" / "Video"
    files_list_root = PROJECT_ROOT / "dataraw" / "folder_file_list"

    resized_keyframe_root = (
        PROJECT_ROOT / "ProcessedData" / "data" / "resized" / "keyframes"
    )
    resized_video_root = PROJECT_ROOT / "ProcessedData" / "data" / "resized" / "video"

    num_threads = get_num_threads(config.resize.num_threads)
    num_image_workers = max(2, num_threads // 2)
    num_video_workers = NVENC_PARALLEL_STREAMS

    IMAGE_BATCH_SIZE = 128

    calculated_image_maxsize = num_image_workers * IMAGE_BATCH_SIZE * 4
    calculated_video_maxsize = max(100, num_video_workers * 1 * 4)

    image_queue = JoinableQueue(maxsize=calculated_image_maxsize)
    video_queue = JoinableQueue(maxsize=calculated_video_maxsize)

    image_workers = [
        Process(
            target=resize_keyframes_worker,
            args=(image_queue, config.resize.frame_height, IMAGE_BATCH_SIZE),
            daemon=True,
        )
        for _ in range(num_image_workers)
    ]
    for p in image_workers:
        p.start()

    video_workers = [
        Process(
            target=encode_video_worker,
            args=(
                video_queue,
                config.resize.video_height,
                config.resize.video_bitrate,
            ),
            daemon=True,
        )
        for _ in range(num_video_workers)
    ]
    for p in video_workers:
        p.start()

    print(f"\n=== Initialized Pipeline | Image workers: {num_image_workers} | Video workers: {num_video_workers} ===")

    for subfolder in sorted(os.listdir(video_root)):
        subfolder_path = video_root / subfolder
        if not subfolder_path.is_dir():
            continue

        print(f"  → Queuing tasks for {subfolder}...")

        videos_path = load_files_list(
            base_dir=video_root / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=None,
        )

        keyframes_src = load_files_list(
            base_dir=keyframe_root / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension="",
        )

        keyframes_dst = load_files_list(
            base_dir=resized_keyframe_root / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension="",
            mkdir=True,
        )

        videos_dst = load_files_list(
            base_dir=resized_video_root / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=".mp4",
            mkdir=True,
        )

        for v_src, v_dst in zip(videos_path, videos_dst):
            wait_for_file(v_src)
            video_queue.put((v_src, v_dst))

        for k_src, k_dst in zip(keyframes_src, keyframes_dst):
            if os.path.isdir(k_src):
                for img_name in sorted(os.listdir(k_src)):
                    src_path = os.path.join(k_src, img_name)
                    dst_path = os.path.join(k_dst, img_name)
                    image_queue.put((src_path, dst_path))

        print(f"  ✓ Queued {subfolder}")

    print(f"\n=== All folders queued. Waiting for workers to finish... ===")

    for _ in range(num_image_workers):
        image_queue.put(None)
    for _ in range(num_video_workers):
        video_queue.put(None)

    image_queue.join()
    video_queue.join()

    for p in image_workers:
        p.join()
    for p in video_workers:
        p.join()

    print(f"✓ All tasks completed successfully.")


if __name__ == "__main__":
    main()
