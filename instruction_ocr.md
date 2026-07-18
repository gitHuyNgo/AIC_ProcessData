# Instruction: preprocess data

Muc tieu: preprocess anh `.webp` trong `D:\AIC\AIC_PREPROCESS\data`, luu ket qua `.webp` vao:

```text
D:\AIC\AIC_PREPROCESS\preprocessed_results\preprocessed
```

Code preprocess chinh nam trong:

```text
D:\AIC\AIC_PREPROCESS\preprocess_data\ocr_preprocess.py
```

De chay nhanh va bo qua anh da preprocess roi, dung runner song song:

```text
D:\AIC\AIC_PREPROCESS\run_preprocess_parallel.py
```

## 1. Lenh preprocess khuyen nghi

Chay preprocess toan bo data:

```powershell
cd D:\AIC\AIC_PREPROCESS

D:\AIC\AIC_Benchmark\ocr_venv\Scripts\python.exe run_preprocess_parallel.py `
  --keyframe-root D:\AIC\AIC_PREPROCESS\data `
  --layout recursive `
  --output-root D:\AIC\AIC_PREPROCESS\preprocessed_results `
  --save-preprocessed `
  --preprocessed-format webp `
  --jpeg-quality 95 `
  --workers 8 `
  --chunksize 8
```

Runner nay mac dinh se skip anh da co output hop le, nen co the chay lai nhieu lan de resume.

## 2. Preprocess mot folder da giai nen tu `.rar`

Khong preprocess truc tiep file `.rar`. Hay preprocess thu muc da giai nen.

Vi du voi:

```text
D:\AIC\AIC_PREPROCESS\data\Videos_K18-004.rar
```

Neu da co thu muc:

```text
D:\AIC\AIC_PREPROCESS\data\Videos_K18-004
```

thi chay:

```powershell
cd D:\AIC\AIC_PREPROCESS

D:\AIC\AIC_Benchmark\ocr_venv\Scripts\python.exe run_preprocess_parallel.py `
  --keyframe-root D:\AIC\AIC_PREPROCESS\data\Videos_K18-004 `
  --layout recursive `
  --output-root D:\AIC\AIC_PREPROCESS\preprocessed_results `
  --save-preprocessed `
  --preprocessed-format webp `
  --jpeg-quality 95 `
  --workers 8 `
  --chunksize 8
```

Neu chua giai nen `.rar`, co the giai nen bang 7-Zip va skip file da ton tai:

```powershell
7z x D:\AIC\AIC_PREPROCESS\data\Videos_K18-004.rar `
  -oD:\AIC\AIC_PREPROCESS\data\Videos_K18-004 `
  -aos
```

Sau do chay lai lenh preprocess folder o tren.

## 3. Y nghia cac tham so quan trong

| Tham so | Gia tri goi y | Y nghia |
| --- | --- | --- |
| `--keyframe-root` | `D:\AIC\AIC_PREPROCESS\data` | Thu muc anh dau vao can preprocess. |
| `--layout` | `recursive` | Quet de quy tat ca anh duoi thu muc dau vao. |
| `--output-root` | `D:\AIC\AIC_PREPROCESS\preprocessed_results` | Thu muc cha cua output. Anh se nam trong `output-root\preprocessed`. |
| `--save-preprocessed` | bat flag nay | Bat luu anh sau preprocess. |
| `--preprocessed-format` | `webp` | Luu output duoi dang `.webp`. |
| `--jpeg-quality` | `95` | Chat luong khi luu WebP/JPEG. |
| `--workers` | `8` | So process chay song song. Tang len neu may con tai nguyen. |
| `--chunksize` | `8` | So anh moi worker nhan trong mot goi viec. |

Output se co dang:

```text
D:\AIC\AIC_PREPROCESS\preprocessed_results\preprocessed\<group_id>\<image_id>.webp
```

Manifest:

```text
D:\AIC\AIC_PREPROCESS\preprocessed_results\selected_keyframes.csv
```

## 4. Goi truc tiep module preprocess trong Python

Neu muon dung truc tiep module `ocr_preprocess.py`, tao `args` nhu sau:

```python
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(r"D:\AIC\AIC_PREPROCESS")
sys.path.insert(0, str(ROOT / "preprocess_data"))

import ocr_preprocess

args = Namespace(
    keyframe_root=ROOT / "data",
    image_root=ROOT / "data",
    output_root=ROOT / "preprocessed_results",
    manifest=None,
    layout="recursive",
    limit=None,
    map_keyframe_root=ocr_preprocess.DEFAULT_MAP_KEYFRAME_ROOT,
    files_list_root=ocr_preprocess.DEFAULT_FILES_LIST_ROOT,
    subfolders=None,
    wait_map_timeout=0.0,
    allow_missing_map=False,
    ignore_selected_column=False,
    max_side=1920,
    min_side=720,
    grayscale=False,
    autocontrast=False,
    autocontrast_cutoff=1.0,
    contrast=1.0,
    sharpness=1.0,
    det_max_side=None,
    det_min_side=None,
    det_autocontrast=True,
    det_autocontrast_cutoff=1.0,
    det_contrast=1.08,
    det_sharpness=1.12,
    save_preprocessed=True,
    preprocessed_format="webp",
    jpeg_quality=95,
)

items = ocr_preprocess.discover_keyframes(
    keyframe_root=args.keyframe_root.resolve(),
    manifest=args.manifest,
    image_root=args.image_root.resolve(),
    limit=args.limit,
    respect_selected_column=not args.ignore_selected_column,
    layout=args.layout,
    map_keyframe_root=args.map_keyframe_root.resolve(),
    files_list_root=args.files_list_root.resolve(),
    subfolders=args.subfolders,
    wait_map_timeout=args.wait_map_timeout,
    allow_missing_map=args.allow_missing_map,
)

prepared_paths = ocr_preprocess.run_preprocessing(items, args)
ocr_preprocess.write_keyframe_manifest(items, prepared_paths, args.output_root.resolve())
```

## 5. GPU

Preprocess anh trong `ocr_preprocess.py` dung PIL, nen buoc resize/autocontrast/sharpness chu yeu chay CPU. GPU chi duoc dung khi chay OCR/model voi `--run-ocr`.

Neu can OCR bang GPU, dung `ocr_preprocess.py` truc tiep:

```powershell
cd D:\AIC\AIC_PREPROCESS

D:\AIC\AIC_Benchmark\ocr_venv\Scripts\python.exe preprocess_data\ocr_preprocess.py `
  --keyframe-root D:\AIC\AIC_PREPROCESS\data `
  --layout recursive `
  --output-root D:\AIC\AIC_PREPROCESS\preprocessed_results `
  --ocr-dir D:\AIC\AIC_Benchmark\ocr\deepdoc_vietocr `
  --save-preprocessed `
  --run-ocr `
  --overwrite `
  --ppocr-device gpu `
  --preprocessed-format webp `
  --jpeg-quality 95
```
