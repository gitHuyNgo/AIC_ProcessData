from pathlib import Path

video_root = Path("dataraw/videos/Video")
output_root = Path("dataraw/folder_file_list")

output_root.mkdir(parents=True, exist_ok=True)

for subfolder in sorted(video_root.iterdir()):
    if not subfolder.is_dir():
        continue

    videos = sorted(
        [f.name for f in subfolder.iterdir() if f.is_file()]
    )

    output_file = output_root / f"files_list_{subfolder.name}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for video in videos:
            f.write(video + "\n")

    print(f"Created {output_file} ({len(videos)} videos)")