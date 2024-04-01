"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
import torchvision.utils as tu

from gsbm.utils import restore_model
from gsbm.dataset import get_dist_boundary

import PIL.Image

import colored_traceback.always
from ipdb import set_trace as debug

import pytorch_lightning as pl

BASE_DIR = Path("outputs/multiruns/afhq")

T = 20  # log_steps


def main(opt, log):

    log(opt)

    ## Load model
    ckpt = BASE_DIR / opt.ckpt
    model, cfg = restore_model(ckpt, device=opt.device)
    model.eval()

    ## Build dataset
    dataset, start_idx, end_idx = build_data(opt, cfg)
    log(f"[Dataset] {opt.transfer=}, {opt.split=}, total size={len(dataset)}!")
    log(
        f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={end_idx-start_idx}!"
    )

    ## Sample Setup
    pl.utilities.seed.seed_everything(opt.seed)

    exp_cfg = f"{ckpt.name[:-5]}_nfe{opt.nfe}_{opt.transfer}_{opt.split}"  # remove ".ckpt" at the end
    sample_dir = BASE_DIR / ckpt.parent.parent / "samples" / exp_cfg
    traj_dir = BASE_DIR / ckpt.parent.parent / "trajs" / exp_cfg
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    log(f"[Sample] Samples will be store in {sample_dir}!")

    ## Sample
    trajs = []
    num = 0
    sample_idx = torch.arange(start_idx, end_idx)
    for idx, batch_idx in enumerate(sample_idx.split(opt.batch)):
        image = torch.stack([dataset[i] for i in batch_idx], dim=0)
        xinit = image.reshape(len(batch_idx), -1).to(opt.device)
        B, D = xinit.shape

        direction = get_direction(opt)
        log(f"[Sample] Sampling ....")
        output = model.sample(
            xinit,
            log_steps=T,
            direction=direction,
            nfe=opt.nfe,
            verbose=opt.verbose and idx == 0,
        )
        xs = output["xs"].detach().cpu()
        assert xs.shape == (B, T, D)

        ## Save images
        gen_image = xs[:, (-1 if direction == "fwd" else 0)].reshape_as(image)
        gen_image_np = (
            (gen_image * 127.5 + 128)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for idx, image_np in zip(batch_idx, gen_image_np):
            image_path = sample_dir / f"{idx:04d}.png"
            PIL.Image.fromarray(image_np, "RGB").save(image_path)

        trajs.append(xs)
        num += B
        log(f"Collected {num} images!")

    ## Save trajs
    all_trajs = torch.cat(trajs, axis=0)
    traj_path = traj_dir / f"p{opt.partition}.pt"
    torch.save(all_trajs, traj_path)
    log("Done!")


def get_direction(opt):
    if opt.transfer == "cat2dog":
        return "fwd"
    elif opt.transfer == "dog2cat":
        return "bwd"
    else:
        raise ValueError()


def get_init_p(opt, cfg):
    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    if opt.transfer[:3] == "cat":
        p = p0 if opt.split == "train" else p0_val
    elif opt.transfer[:3] == "dog":
        p = p1 if opt.split == "train" else p1_val
    else:
        raise ValueError()
    return p


def build_partition(opt, full_dataset):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    # assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx + 1) * n_samples_per_part

    if part_idx == (n_part - 1):
        end_idx = n_samples

    return start_idx, end_idx


def build_data(opt, cfg):
    pinit = get_init_p(opt, cfg)
    dataset = pinit.dataset
    start_idx, end_idx = build_partition(opt, dataset)
    return dataset, start_idx, end_idx


if __name__ == "__main__":
    """
        python afhq_sample.py --ckpt 2023.09.09/163416/0/checkpoints/epoch-015_step-400000.ckpt \
            --transfer cat2dog --partition 0_4 --nfe 1000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nfe", type=int, default=1000)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--transfer", type=str, default=None, choices=["cat2dog", "dog2cat"]
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--partition", type=str, default="0_1")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")

    opt = parser.parse_args()

    assert opt.nfe > T

    def do_nothing(*arg):
        return

    log = print if opt.verbose else do_nothing
    main(opt, log)
