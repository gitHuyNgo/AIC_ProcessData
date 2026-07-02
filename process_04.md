

```bash
mkdir -p data/zip
```

```bash
rclone copy aic:videos ./data/zip -P --transfers 16 --checkers 16
```

```bash
mkdir -p dataraw/videos/Video
```

```bash
for zip_file in data/zip/*.zip; do
  folder_name=$(basename "$zip_file" .zip)
  mkdir -p "dataraw/videos/Video/$folder_name"
  unzip -q "$zip_file" -d "dataraw/videos/Video/$folder_name"
done
```

```bash
rm -rf data # use this if you want to delete raw downloaded data to free up space
```

```bash
mkdir -p ProcessedData/data/keyframes
```

```bash
rclone copy aic:keyframes ProcessedData/data/keyframes -P --transfers 16 --checkers 16
```

```bash
for zip_file in ProcessedData/data/keyframes/*.zip; do
  folder_name=$(basename "$zip_file" .zip)
  mkdir -p "ProcessedData/data/keyframes/$folder_name"
  echo "Unzipping $folder_name..."
  unzip -q "$zip_file" -d "ProcessedData/data/keyframes/$folder_name" && rm "$zip_file"
done
```

```bash
mkdir -p dataraw/folder_file_list
```

```bash
rclone copy aic:folder_file_list dataraw/folder_file_list -P --transfers 16 --checkers 16
```





```bash
cd AIC_ProcessData
pip install -r src/requirements/resize_requirements.txt
PYTHONPATH=src python3 -m src.resize
```

```bash
cd ..
mkdir -p ProcessedData/data/map_keyframes
rclone copy aic:map_keyframes ProcessedData/data/map_keyframes -P --transfers 16 --checkers 16
cd AIC_ProcessData
pip install -r src/requirements/asr_requirements.txt
```

```bash
find /usr -name "libcublas.so.12" 2>/dev/null # run this if show nothing then continue run those below in this box
sudo apt update
sudo apt install cuda-toolkit-12-8
```

```bash
PYTHONPATH=src python3 -m src.asr_video
```


```bash
cd ProcessedData/data/
zip -r asr_transcript.zip asr_transcript/
rclone copy asr_transcript.zip aic:embeddings/
```

```bash
cd ProcessedData/data/resized/keyframes

for dir in */; do
    dir_name=${dir%/}
    echo "--- Processing: $dir_name ---"
    zip -r "${dir_name}.zip" "$dir_name"
    rclone copy "${dir_name}.zip" aic:resized/keyframes/ -P
    rm "${dir_name}.zip"
done
```

```bash
cd ProcessedData/data/resized/video

for dir in */; do
    dir_name=${dir%/}
    echo "--- Processing: $dir_name ---"
    zip -r "${dir_name}.zip" "$dir_name"
    rclone copy "${dir_name}.zip" aic:resized/video/ -P
    rm "${dir_name}.zip"
done
```