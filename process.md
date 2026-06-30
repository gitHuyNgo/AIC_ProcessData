# AIC_ProcessData Process

This document describes the end-to-end process for preparing `.mp4` video data and running the full processing pipeline.

## 1. Install System Tools

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install ffmpeg unzip tmux -y
```

If you need to download data from cloud storage with `rclone`:

```bash
sudo -v
curl https://rclone.org/install.sh | sudo bash
rclone config
```

Make sure `ffmpeg` and `ffprobe` are available:

```bash
ffmpeg -version
ffprobe -version
```

## 2. Create Python Environment

Create and activate a Conda environment:

```bash
conda create -n aic_process python=3.10 -y
conda activate aic_process
```

Install Python dependencies:

```bash
pip install -r src/requirements/requirements_encode_video.txt
pip install -r src/requirements/requirements_scene_boundary.txt
pip install -r src/requirements/requirements_save_keyframes.txt
```

## 3. Prepare Input Videos

Place all input `.mp4` videos under:

```txt
dataraw/videos/Video/
```

Recommended layout:

```txt
dataraw/videos/Video/L01/video_001.mp4
dataraw/videos/Video/L01/video_002.mp4
dataraw/videos/Video/L02/video_003.mp4
```

The pipeline scans `.mp4` files recursively, so nested folders are allowed.

## 4. Run The Full Pipeline

Run all steps in order:

```bash
python src/pipeline.py --device cuda
```

The pipeline executes:

1. `encode_video`: creates frame embeddings.
2. `scene_boundary`: detects scene boundaries.
3. `clustering`: selects keyframe indices.
4. `save_keyframes`: saves keyframe images and mapping files.

To test with only a few videos:

```bash
python src/pipeline.py --device cuda --limit-videos 2
```

To use CPU:

```bash
python src/pipeline.py --device cpu
```

To process videos from a different folder:

```bash
python src/pipeline.py --video-root path/to/mp4_folder --device cuda
```

To rerun and overwrite existing outputs:

```bash
python src/pipeline.py --device cuda --overwrite
```

## 5. Output Structure

After the pipeline finishes, outputs are saved to:

```txt
dataraw/embeddings/...                         # frame embeddings .npy
ProcessedData/scence_boundary/...              # scene boundary .txt
dataraw/embeddings/keyframes_indices_B32_*/... # selected keyframe indices .txt
ProcessedData/data/keyframes/...               # keyframe .webp files
ProcessedData/data/map_keyframes/...           # keyframe mapping .csv files
```

## 6. Notes

- The pipeline only processes `.mp4` files.
- `sample_every` is fixed at `1` inside the pipeline so keyframe indices match the original video frame numbers.
- If decoding fails for some codecs, install `ffmpeg`/`ffprobe` and make sure they are in `PATH`.
- If CUDA is unavailable, run with `--device cpu`.
