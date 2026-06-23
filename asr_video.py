import sys
from pathlib import Path
import os
import csv
import json
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import numpy as np
from queue import Queue
from threading import Thread
import concurrent.futures
import subprocess
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "config"

from faster_whisper import WhisperModel, BatchedInferencePipeline
from utils import wait_for_file


def detect_optimal_hardware():
    physical_cores = os.cpu_count() or 1
    try:
        import torch
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_gpu_workers = math.floor(total_vram_gb / 8.0)
            if max_gpu_workers < 1:
                max_gpu_workers = 1
            max_gpu_workers = min(max_gpu_workers, physical_cores)
            cpu_threads_per_worker = max(1, min(4, physical_cores // max_gpu_workers))
            return "cuda", max_gpu_workers, cpu_threads_per_worker
    except ImportError:
        pass
    
    return "cpu", 1, physical_cores


def extract_audio_optimized(video_path, wav_path):
    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# ================= ASR =================
def run_asr(audio_path, model, language=None):

    temperature_schedule = [0.2, 0.4, 0.6]

    # Process dynamically and rely on Faster-Whisper's internal per-segment fallback.
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        best_of=5,
        temperature=temperature_schedule,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.75,
        no_speech_threshold=0.65,
        batch_size=16,
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.35,
            min_speech_duration_ms=500,
            min_silence_duration_ms=800,
            speech_pad_ms=300,
        ),
    )

    cleaned_segments = []

    for seg in segments_gen:
        text = seg.text.strip()

        if len(text) < 3:
            continue

        if hasattr(seg, "avg_logprob") and seg.avg_logprob < -1.2:
            continue

        if hasattr(seg, "no_speech_prob") and seg.no_speech_prob > 0.65:
            continue

        # Append pure float values directly (Removing redundant type casting)
        cleaned_segments.append(
            (seg.start, seg.end, text)
        )

    print(f"Detected {len(cleaned_segments)} segments in {audio_path}")

    return cleaned_segments


def load_map_csv(path):
    # Return two aligned NumPy arrays (pts_times and frame_indices)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pts_times = []
        frame_indices = []
        for row in reader:
            pts_times.append(float(row["pts_time"]))
            frame_indices.append(int(row["frame_idx"]))
            
    pts_times = np.array(pts_times, dtype=np.float64)
    frame_indices = np.array(frame_indices, dtype=np.int64)
    
    # Ensure pts_times is sorted
    sort_idx = np.argsort(pts_times)
    return pts_times[sort_idx], frame_indices[sort_idx]


def build_asr_transcript(cleaned_segments, asr_csv_path, map_csv_path, output_json_path):
    # Apply the string formatting strictly at the very end when writing the rows to the CSV
    formatted_segments = [
        (f"{seg[0]:.3f}", f"{seg[1]:.3f}", seg[2]) for seg in cleaned_segments
    ]
    # Consumer thread handles the disk writes for CSV as well to keep Producer completely free of disk writes
    with open(asr_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["start", "end", "text"])
        writer.writerows(formatted_segments)

    if not Path(map_csv_path).exists():
        return

    pts_times, frame_indices = load_map_csv(map_csv_path)

    results = []

    if cleaned_segments:
        # Fully Vectorize NumPy Arrays (Zero Python Loops for Searching)
        starts_arr = np.array([seg[0] for seg in cleaned_segments], dtype=np.float64)
        ends_arr = np.array([seg[1] for seg in cleaned_segments], dtype=np.float64)
        
        # Perform search simultaneously for all segments in pure C-level NumPy
        start_idxs = np.searchsorted(pts_times, starts_arr, side='left')
        end_idxs = np.searchsorted(pts_times, ends_arr, side='right')
        
        for i, seg in enumerate(cleaned_segments):
            start_val, end_val, text = seg
            imgs = frame_indices[start_idxs[i]:end_idxs[i]].tolist()
            
            results.append({
                "start": start_val,
                "end": end_val,
                "text": text,
                "imgs": imgs,
            })

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(config: DictConfig):
    print(os.getcwd())
    video_root = PROJECT_ROOT / "ProcessedData" / "data" / "resized" / "video"
    files_list_root = PROJECT_ROOT / "ProcessedData" / "folder_file_list"
    asr_csv_root = PROJECT_ROOT / "ProcessedData" / "data" / "asr_segments"
    map_root = PROJECT_ROOT / "ProcessedData" / "data" / "map_keyframes"
    asr_transcript_root = PROJECT_ROOT / "ProcessedData" / "data" / "asr_transcript"
    wav_root = PROJECT_ROOT / "ProcessedData" / "data" / "wavs"

    # Eliminate Redundant File List Parsing (Global Task List)
    tasks_list = []
    if os.path.exists(video_root):
        for subfolder in sorted(os.listdir(video_root)):
            subfolder_path = os.path.join(video_root, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            files_list_path = files_list_root / f"files_list_{subfolder}.txt"
            if not files_list_path.exists():
                continue
                
            with open(files_list_path, "r", encoding="utf-8") as f:
                base_names = [line.strip() for line in f if line.strip()]

            (asr_csv_root / subfolder).mkdir(parents=True, exist_ok=True)
            (map_root / subfolder).mkdir(parents=True, exist_ok=True)
            (asr_transcript_root / subfolder).mkdir(parents=True, exist_ok=True)
            (wav_root / subfolder).mkdir(parents=True, exist_ok=True)

            for base_name in base_names:
                name_stem = Path(base_name).stem
                vp = str(Path(subfolder_path) / base_name)
                wav_path = str(wav_root / subfolder / f"{name_stem}.wav")
                out_csv = str(asr_csv_root / subfolder / f"{name_stem}.csv")
                map_csv = str(map_root / subfolder / f"{name_stem}.csv")
                out_json = str(asr_transcript_root / subfolder / f"{name_stem}.json")
                tasks_list.append((vp, wav_path, out_csv, map_csv, out_json))

    # --- PHASE 1: AUDIO EXTRACTION (FFmpeg) ---
    print("\n--- PHASE 1: AUDIO EXTRACTION (FFmpeg) ---")
    def process_ffmpeg(task):
        vp_path, wav_path, out_csv, map_csv, out_json = task
        if not os.path.exists(wav_path):
            wait_for_file(vp_path)
            try:
                extract_audio_optimized(vp_path, wav_path)
            except Exception:
                return None
        return task

    valid_tasks = []
    safe_io_workers = min(os.cpu_count() or 1, 8) 
    with concurrent.futures.ThreadPoolExecutor(max_workers=safe_io_workers) as executor:
        results = list(tqdm(executor.map(process_ffmpeg, tasks_list), total=len(tasks_list), desc="Audio Extraction"))
        valid_tasks = [r for r in results if r is not None]
        
    if not valid_tasks:
        print("All audio extraction failed. Stopping.")
        return

    # --- DYNAMIC HARDWARE CONFIGURATION ---
    device_mode, gpu_workers, cpu_threads_per_worker = detect_optimal_hardware()
    print("\n--- DYNAMIC HARDWARE CONFIGURATION ---")
    print(f"Device: {device_mode.upper()}")
    print(f"Max GPU Workers: {gpu_workers}")
    print(f"CPU Threads Per Worker: {cpu_threads_per_worker}")

    model = WhisperModel(
        config.asr.model_size,
        device=device_mode,
        compute_type=config.asr.compute_type,
        num_workers=gpu_workers,
        cpu_threads=cpu_threads_per_worker,
    )

    batched_model = BatchedInferencePipeline(model=model)

    # --- PHASE 2: ASR INFERENCE (Whisper) ---
    print("\n--- PHASE 2: ASR INFERENCE (Whisper) ---")
    
    # Setup Asynchronous Queue with Backpressure Safety Mechanism
    task_queue = Queue(maxsize=8)
    
    # Initialize a single tqdm instance using len(tasks_list)
    pbar = tqdm(total=len(valid_tasks), desc="Pipeline Progress")

    def consumer_worker():
        # Consumer Thread (CPU/IO Bound): Monitors the Queue and executes build_asr_transcript
        while True:
            task = task_queue.get()
            if task is None: # Sentinel/poison-pill mechanism
                task_queue.task_done()
                break
            cleaned_segments, out_csv, map_csv, out_json = task
            build_asr_transcript(cleaned_segments, out_csv, map_csv, out_json)
            
            pbar.update(1)
            task_queue.task_done()

    def producer_worker():
        with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_workers) as executor:
            futures = []
            
            for vp, wav_path, out_csv, map_csv, out_json in valid_tasks:
                def process_and_queue(audio_path=wav_path, csv_out=out_csv, map_out=map_csv, json_out=out_json):
                    
                    # ---- ASR ----
                    cleaned_segments = run_asr(
                        audio_path=audio_path,
                        model=batched_model,
                        language=config.asr.language,
                    )
                    
                    # Thread-safe queue push directly to the consumer
                    task_queue.put((cleaned_segments, csv_out, map_out, json_out))

                futures.append(executor.submit(process_and_queue))

            # Ensure any exceptions within threads are safely propagated
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during ASR processing: {e}")

        # Send poison-pill to gracefully terminate the consumer AFTER thread pool finishes
        task_queue.put(None)

    consumer_thread = Thread(target=consumer_worker, daemon=True)
    producer_thread = Thread(target=producer_worker)

    consumer_thread.start()
    producer_thread.start()

    # The main thread safely joins and waits for the entire pipeline
    producer_thread.join()
    task_queue.join() # Wait for all tasks in queue to be processed
    consumer_thread.join()
    
    # Close progress bar properly after all threads joined
    pbar.close()


if __name__ == "__main__":
    main()