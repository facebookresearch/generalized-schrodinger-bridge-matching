"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import math
import glob
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("[afhq preprocessing]")


class NumpyFolder(Dataset):
    def __init__(self, folder, resize):
        files_grabbed = []
        for tt in ("*.png", "*.jpg"):
            files_grabbed.extend(glob.glob(os.path.join(folder, tt)))
        files_grabbed = sorted(files_grabbed)

        self.img_paths = files_grabbed

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_set = Image.open(self.img_paths[idx])
        image_set = self.transform(image_set)
        image_tensor = np.asarray(image_set)
        return image_tensor


def main(opt):
    for split in ("train", "val"):
        for ani in ("cat", "dog", "wild"):
            log.info(f"Extracting {split=}, {ani=}, resolution={opt.resolution} ...")

            dataset = NumpyFolder(opt.dir / f"{split}/{ani}", resize=opt.resolution)
            np_imgs = np.stack([dataset[ii] for ii in range(len(dataset))], axis=0)

            os.makedirs(opt.save, exist_ok=True)
            fn = opt.save / f"afhq{opt.resolution}-{split}-{ani}.npz"
            np.savez(fn, data=np_imgs)
            log.info(f"Saved in {fn=}!")


if __name__ == "__main__":
    """
    # PATH_TO_AFHQ="../stargan-v2/data/afhq"
    python afhq_preprocess.py --dir $PATH_TO_AFHQ
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--save", type=Path, default="data/")
    parser.add_argument("--resolution", type=int, default=64)

    opt = parser.parse_args()
    main(opt)
