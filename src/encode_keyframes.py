import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "config"

import torch
from PIL import Image

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
import open_clip

from utils import load_files_list, wait_for_file

import csv

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def load_images_id(map_kf_path):
    with open(map_kf_path, newline="", mode="r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        header_idx = header.index("id")
        ids = [row[header_idx] for row in reader]
    return ids


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


class KF_loader(Dataset):
    def __init__(self, map_kf_path, kf_path, img_preprocess):
        self.kf_path = kf_path
        self.images_id = load_images_id(map_kf_path)
        self.img_preprocess = img_preprocess

    def __len__(self):
        return len(self.images_id)

    def __getitem__(self, idx):
        img_id = self.images_id[idx]
        img_path = os.path.join(self.kf_path, f"{int(img_id):06d}.webp")
        img = Image.open(img_path)
        return self.img_preprocess(img)


# ──────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────


class Encoder:
    def __init__(self, model_type, model_arch, model_pretrained, jit):
        self.model_type = model_type

        if model_type == "clip":
            model, _, preprocess_fn = open_clip.create_model_and_transforms(
                model_arch,
                pretrained=model_pretrained,
                jit=jit,
            )
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            model = model.cuda()

            self.model = model
            self._preprocess_fn = preprocess_fn


        else:
            raise ValueError(
                f"model_type không hợp lệ: '{model_type}'. Dùng 'clip'."
            )

    def preprocess(self, img):
        return self._preprocess_fn(img)

    def encode_batch(self, batch, normalization):
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):

                image_features = self.model.encode_image(batch)

            if normalization:
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

        return image_features.cpu().numpy()


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────


def encode_keyframes(
    kf_path,
    map_kf_path,
    output_path,
    model,
    batch_size,
    num_workers,
    normalization,
):
    dataset = KF_loader(map_kf_path, kf_path, model.preprocess)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dl = DataLoader(
        dataset,
        **loader_kwargs,
    )

    emb_file = None
    offset = 0

    for batch in tqdm(dl, leave=False):
        batch = batch.cuda(non_blocking=True)
        image_features = model.encode_batch(batch, normalization)

        if emb_file is None:
            emb_file = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=image_features.dtype,
                shape=(len(dataset), image_features.shape[-1]),
            )

        next_offset = offset + image_features.shape[0]
        emb_file[offset:next_offset] = image_features
        offset = next_offset

    if emb_file is not None:
        emb_file.flush()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(config: DictConfig):

    video_root = PROJECT_ROOT / "dataraw" / "videos" / "Video"
    files_list_root = PROJECT_ROOT / "dataraw" / "folder_file_list"

    print(f"model type : {config.key_model.model_type}")
    print(f"model arch : {config.key_model.model_arch}")
    print(f"pretrain   : {config.key_model.model_pretrained}")
    print(f"batch size : {config.key_model.batch_size}")
    if not config.key_model.normalization:
        print("⚠️  no normalization")

    model_tag = HydraConfig.get().runtime.choices["key_model"]
    print(f"output tag : {model_tag}")

    enc = Encoder(
        model_type=config.key_model.model_type,
        model_arch=config.key_model.model_arch,
        model_pretrained=config.key_model.model_pretrained,
        jit=config.key_model.jit,
    )

    for subfolder in sorted(os.listdir(video_root)):
        subfolder_path = video_root / subfolder
        if not subfolder_path.is_dir():
            continue

        print(f"\n📂 Processing subfolder: {subfolder}")

        keyframes_path = load_files_list(
            base_dir=PROJECT_ROOT / "ProcessedData" / "data" / "keyframes" / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension="",
        )

        map_keyframes_path = load_files_list(
            base_dir=PROJECT_ROOT
            / "ProcessedData"
            / "data"
            / "map_keyframes"
            / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=".csv",
        )

        output_paths = load_files_list(
            base_dir=PROJECT_ROOT
            / "ProcessedData"
            / "data"
            / "embeddings"
            / model_tag
            / subfolder,
            files_list_path=files_list_root / f"files_list_{subfolder}.txt",
            with_extension=".npy",
            mkdir=True,
        )

        for kfp, mkfp, outp in tqdm(
            zip(keyframes_path, map_keyframes_path, output_paths),
            total=len(output_paths),
            desc=subfolder,
        ):
            wait_for_file(mkfp)

            if os.path.exists(outp):
                continue

            encode_keyframes(
                kf_path=kfp,
                map_kf_path=mkfp,
                output_path=outp,
                model=enc,
                batch_size=config.key_model.batch_size,
                num_workers=config.key_model.num_workers,
                normalization=config.key_model.normalization,
            )


if __name__ == "__main__":
    main()
