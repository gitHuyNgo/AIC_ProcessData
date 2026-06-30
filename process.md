# AIC_ProcessData Process

This document describes how to download zipped video data from Google Drive, extract the `.mp4` files, and run the full processing pipeline.

## 1. Install System Tools

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install ffmpeg unzip zip tmux curl -y
```

Install `rclone` for Google Drive downloads:

```bash
sudo -v
curl https://rclone.org/install.sh | sudo bash
```

Check that the required tools are available:

```bash
ffmpeg -version
ffprobe -version
rclone version
```

## 2. Configure Google Drive Access

Run:

```bash
rclone config
```

Create a Google Drive remote. In the examples below, the remote name is:

```txt
gdrive
```

If you choose another remote name, replace `gdrive` in the commands below.

## 3. Download Zip Files From Google Drive

Your data is stored on Google Drive as multiple `.zip` files. Each zip file contains many `.mp4` videos.

Create a local folder for the zip files:

```bash
mkdir -p data/zip
```

If the shared folder appears in rclone as `AIC_2026/videos`, download all zip files with:

```bash
rclone copy "gdrive:AIC_2026/videos" data/zip --drive-shared-with-me --include "*.zip" -P
```

If rclone cannot find the shared folder by name, use the Google Drive folder id from the shared link:

```bash
rclone copy gdrive: data/zip --drive-shared-with-me --drive-root-folder-id 1hsybk0yYP8xpkwpNRDvOUK-0nRNtnM-K --include "*.zip" -P
```

Expected local files:

```txt
data/zip/Videos_K16.zip
data/zip/Videos_K17.zip
data/zip/Videos_K18.zip
...
```

## 4. Extract Zip Files

The pipeline reads `.mp4` files from:

```txt
dataraw/videos/Video/
```

Create that folder:

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

After extraction, the layout should look like:

```txt
dataraw/videos/Video/Videos_K16/*.mp4
dataraw/videos/Video/Videos_K17/*.mp4
dataraw/videos/Video/Videos_K18/*.mp4
```

The pipeline scans `.mp4` files recursively, so it is fine if the zip files contain nested folders.

## 5. Create Python Environment

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

## 6. Run The Full Pipeline

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

To rerun and overwrite existing outputs:

```bash
python src/pipeline.py --device cuda --overwrite
```

## 7. Output Structure

After the pipeline finishes, outputs are saved to:

```txt
dataraw/embeddings/...                         # frame embeddings .npy
ProcessedData/scence_boundary/...              # scene boundary .txt
dataraw/embeddings/keyframes_indices_B32_*/... # selected keyframe indices .txt
ProcessedData/data/keyframes/...               # keyframe .webp files
ProcessedData/data/map_keyframes/...           # keyframe mapping .csv files
```

## 8. Zip And Upload Outputs To Google Drive

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

Upload the output zip back to Google Drive:

```bash
rclone copy data/output "gdrive:AIC_2026/outputs" -P
```

If you need to upload to a shared Google Drive folder by folder id, use:

```bash
rclone copy data/output gdrive: --drive-shared-with-me --drive-root-folder-id YOUR_OUTPUT_FOLDER_ID -P
```

Verify that the files were uploaded:

```bash
rclone ls "gdrive:AIC_2026/outputs"
```

## 9. Notes

- The input data starts as `.zip` files on Google Drive.
- Each zip file can contain many `.mp4` videos.
- The pipeline only processes `.mp4` files.
- `sample_every` is fixed at `1` inside the pipeline so keyframe indices match the original video frame numbers.
- If decoding fails for some codecs, confirm that `ffmpeg` and `ffprobe` are installed and available in `PATH`.
- If CUDA is unavailable, run with `--device cpu`.
- The final output zip includes embeddings, scene boundaries, selected keyframe indices, keyframe images, and mapping files.
