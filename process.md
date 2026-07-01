# AIC_ProcessData Full Pipeline

This guide describes the full workflow for running the project on VastAI:

1. Clone the repository.
2. Install system tools and Python dependencies.
3. Download zipped `.mp4` data from Google Drive.
4. Extract videos into the expected input folder.
5. Run the processing pipeline in phases.
6. Zip all outputs.
7. Upload the final zip files back to Google Drive.

## 1. Start A VastAI Instance

Use a CUDA image with Python and PyTorch if possible.

After connecting to the instance, check the GPU:

```bash
nvidia-smi
```

Start a `tmux` session so the job keeps running if your SSH connection drops:

```bash
tmux new -s aic
```

To detach from tmux:

```bash
Ctrl+b
d
```

To reconnect later:

```bash
tmux attach -t aic
```

## 2. Clone The Repository

Clone the project:

```bash
git clone <YOUR_REPO_URL>
cd AIC_ProcessData
```

If the repository is already cloned, update it:

```bash
cd AIC_ProcessData
git pull
```

## 3. Install System Tools

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install git ffmpeg unzip zip tmux curl -y
```

Install `rclone` for Google Drive download/upload:

```bash
sudo -v
curl https://rclone.org/install.sh | sudo bash
```

Check the tools:

```bash
ffmpeg -version
ffprobe -version
rclone version
```

## 4. Create Python Environment

If Conda is available:

```bash
conda create -n aic_process python=3.10 -y
conda activate aic_process
```

If Conda is not available, use `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r src/requirements/requirements_encode_video.txt
pip install -r src/requirements/requirements_scene_boundary.txt
pip install -r src/requirements/requirements_save_keyframes.txt
```

Verify CUDA from Python:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. Configure Google Drive With Rclone

Run:

```bash

```

Create a Google Drive remote. This guide uses:

```txt
aic
```

If you use another remote name, replace `aic` in the commands below.

## 6. Download Zip Data From Google Drive

Your input data is on Google Drive as `.zip` files. Each zip file contains many `.mp4` videos.

Create a local folder for downloaded zip files:

```bash
mkdir -p data/zip
```

Check available disk space before downloading:

```bash
df -h
du -sh data/zip 2>/dev/null || true
```

First, list the files inside the shared Google Drive path:

```bash
rclone lsf "aic:AIC_2026/videos" \
  --drive-shared-with-me \
  --files-only \
  --max-depth 1
```

Do a dry run before downloading:

```bash
rclone copy "aic:AIC_2026/videos" data/zip \
  --drive-shared-with-me \
  --files-only \
  --max-depth 1 \
  --filter "+ /Videos_*.zip" \
  --filter "- *" \
  --dry-run \
  -P
```

If the dry run only shows the expected video zip files and the disk has enough free space, download them:

```bash
rclone copy "aic:AIC_2026/videos" data/zip \
  --drive-shared-with-me \
  --files-only \
  --max-depth 1 \
  --filter "+ /Videos_*.zip" \
  --filter "- *" \
  -P
```

If the instance does not have enough disk space for all zip files at once, download one zip file at a time:

```bash
rclone copy "aic:AIC_2026/videos" data/zip \
  --drive-shared-with-me \
  --files-only \
  --max-depth 1 \
  --filter "+ /Videos_K17.zip" \
  --filter "- *" \
  -P
```

If a download fails with `no space left on device`, remove failed partial files before retrying:

```bash
find data/zip -type f -name "*.partial" -print
find data/zip -type f -name "*.partial" -delete
```

Expected files:

```txt
Videos_K16.zip
Videos_K17.zip
Videos_K18.zip
Videos_K19.zip
Videos_K20.zip
Videos_L25_a1.zip
Videos_L28_a.zip
Videos_L29_a.zip
Videos_L30_a.zip
```

Check downloaded files:

```bash
ls -lh data/zip
```

Expected files:

```txt
data/zip/Videos_K16.zip
data/zip/Videos_K17.zip
data/zip/Videos_K18.zip
data/zip/Videos_K19.zip
...
```

## 7. Extract Videos

The pipeline reads input videos from:

```txt
dataraw/videos/Video/
```

Create the input folder:

```bash
mkdir -p dataraw/videos/Video
```

Extract each zip file into its own subfolder:

```bash
for zip_file in data/zip/*.zip; do
  folder_name=$(basename "$zip_file" .zip)
  mkdir -p "dataraw/videos/Video/$folder_name"
  unzip -q "$zip_file" -d "dataraw/videos/Video/$folder_name"
done
```

If disk space is limited, extract one zip file and then remove that zip file:

```bash
zip_file="data/zip/Videos_K17.zip"
folder_name=$(basename "$zip_file" .zip)
mkdir -p "dataraw/videos/Video/$folder_name"
unzip -q "$zip_file" -d "dataraw/videos/Video/$folder_name"
rm "$zip_file"
```

Check extracted videos:

```bash
find dataraw/videos/Video -type f -name "*.mp4" | head
find dataraw/videos/Video -type f -name "*.mp4" | wc -l
```

Example layout:

```txt
dataraw/videos/Video/Videos_K16/*.mp4
dataraw/videos/Video/Videos_K17/*.mp4
dataraw/videos/Video/Videos_K18/*.mp4
```

The pipeline scans `.mp4` files recursively, so nested folders inside the zip files are allowed.

## 8. Test The Pipeline

Before running everything, test the same phase order on two videos:

```bash
python src/pipeline.py --device cuda --limit-videos 2 --overwrite --skip-scene-boundary --skip-clustering --skip-save-keyframes
python src/pipeline.py --device cuda --limit-videos 2 --overwrite --skip-encode --skip-clustering --skip-save-keyframes
python src/pipeline.py --device cuda --limit-videos 2 --overwrite --skip-encode --skip-scene-boundary --skip-save-keyframes
python src/pipeline.py --device cuda --limit-videos 2 --overwrite --skip-encode --skip-scene-boundary --skip-clustering
```

If CUDA is not available, use CPU:

```bash
python src/pipeline.py --device cpu --limit-videos 2 --overwrite --skip-scene-boundary --skip-clustering --skip-save-keyframes
python src/pipeline.py --device cpu --limit-videos 2 --overwrite --skip-encode --skip-clustering --skip-save-keyframes
python src/pipeline.py --device cpu --limit-videos 2 --overwrite --skip-encode --skip-scene-boundary --skip-save-keyframes
python src/pipeline.py --device cpu --limit-videos 2 --overwrite --skip-encode --skip-scene-boundary --skip-clustering
```

The four commands run:

1. `encode_video` only.
2. `scene_boundary` only.
3. `clustering` only.
4. `save_keyframes` only.

## 9. Run The Full Pipeline In Phases

Run all videos in this order. This keeps the heavy ViT-H-14 model loaded only during the encode phase, then runs the later CPU/IO-heavy phases separately.

### 9.1 Encode All Videos

Create frame embeddings for every video:

```bash
python src/pipeline.py --device cuda \
  --skip-scene-boundary \
  --skip-clustering \
  --skip-save-keyframes
```

### 9.2 Detect Scene Boundaries

Run PySceneDetect for every video after embeddings are done:

```bash
python src/pipeline.py --device cuda \
  --skip-encode \
  --skip-clustering \
  --skip-save-keyframes \
  --scene-downscale 4
```

Use `--scene-downscale 1` if you want full-resolution PySceneDetect behavior. Larger values are faster but can slightly change scene boundaries.

### 9.3 Select Keyframe Indices

Run clustering after scene boundary files exist:

```bash
python src/pipeline.py --device cuda \
  --skip-encode \
  --skip-scene-boundary \
  --skip-save-keyframes
```

### 9.4 Save Keyframe Images

Save full-resolution `.webp` keyframes after keyframe index files exist:

```bash
python src/pipeline.py --device cuda \
  --skip-encode \
  --skip-scene-boundary \
  --skip-clustering
```

The saved keyframes keep the original video resolution.

### 9.5 Save Keyframes In Parallel

The save-keyframes phase can be sharded across multiple processes because it does not load the ViT-H-14 model:

```bash
SHARDS=4

for i in $(seq 0 $((SHARDS - 1))); do
  python src/pipeline.py --device cuda \
    --skip-encode \
    --skip-scene-boundary \
    --skip-clustering \
    --num-shards "$SHARDS" \
    --shard-index "$i" &
done

wait
```

If CPU, disk, or WebP encoding becomes overloaded, use `SHARDS=2`.

### 9.6 Rerun A Phase

To rerun and overwrite outputs for a phase, add `--overwrite` to that phase command. For example, rerun only clustering:

```bash
python src/pipeline.py --device cuda \
  --overwrite \
  --skip-encode \
  --skip-scene-boundary \
  --skip-save-keyframes
```

### 9.7 Resume After A Stop

If the job is interrupted, run the same phase command again **without** `--overwrite`. The pipeline skips files that already exist and continues with the unfinished videos.

For example, resume encode:

```bash
python src/pipeline.py --device cuda \
  --skip-scene-boundary \
  --skip-clustering \
  --skip-save-keyframes
```

Resume scene boundary:

```bash
python src/pipeline.py --device cuda \
  --skip-encode \
  --skip-clustering \
  --skip-save-keyframes \
  --scene-downscale 4
```

Resume clustering:

```bash
python src/pipeline.py --device cuda \
  --skip-encode \
  --skip-scene-boundary \
  --skip-save-keyframes
```

Resume save keyframes:

```bash
python src/pipeline.py --device cuda \
  --skip-encode \
  --skip-scene-boundary \
  --skip-clustering
```

Only use `--overwrite` when you intentionally want to regenerate existing outputs.

## 10. Output Structure

After the pipeline finishes, outputs are saved to:

```txt
dataraw/embeddings/...                         # frame embeddings .npy
ProcessedData/scence_boundary/...              # scene boundary .txt
dataraw/embeddings/keyframes_indices_B32_*/... # selected keyframe indices .txt
ProcessedData/data/keyframes/...               # keyframe .webp files
ProcessedData/data/map_keyframes/...           # keyframe mapping .csv files
```

Quick output checks:

```bash
find dataraw/embeddings -type f -name "*.npy" | wc -l
find ProcessedData/scence_boundary -type f -name "*.txt" | wc -l
find ProcessedData/data/keyframes -type f -name "*.webp" | wc -l
find ProcessedData/data/map_keyframes -type f -name "*.csv" | wc -l
```

## 11. Zip All Outputs

Create a folder for final zip artifacts:

```bash
mkdir -p data/output
```

Zip all pipeline outputs:

```bash
zip -r data/output/AIC_ProcessData_outputs.zip \
  dataraw/embeddings \
  ProcessedData
```

If the output is very large, create split zip files with 4 GB parts:

```bash
zip -r -s 4g data/output/AIC_ProcessData_outputs_split.zip \
  dataraw/embeddings \
  ProcessedData
```

Check the zip files:

```bash
ls -lh data/output
```

## 12. Upload Outputs Back To Google Drive

Upload the output zip folder:

```bash
rclone copy data/output "aic:AIC_2026/outputs" -P
```

If you need to upload to a shared Drive folder by folder id:

```bash
rclone copy data/output aic: \
  --drive-shared-with-me \
  --drive-root-folder-id YOUR_OUTPUT_FOLDER_ID \
  -P
```

Verify uploaded files:

```bash
rclone ls "aic:AIC_2026/outputs"
```

## 13. Notes

- Input data starts as `.zip` files on Google Drive.
- Each zip file can contain many `.mp4` videos.
- The pipeline only processes `.mp4` files.
- `sample_every` is fixed at `1` inside the pipeline so keyframe indices match the original video frame numbers.
- If video decoding fails, confirm that `ffmpeg` and `ffprobe` are installed and available in `PATH`.
- If CUDA is unavailable, run with `--device cpu`.
- The final output zip includes embeddings, scene boundaries, selected keyframe indices, keyframe images, and mapping files.
