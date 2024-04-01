"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Union, Dict, Any
import os
from glob import glob
import numpy as np
import importlib
from omegaconf import OmegaConf
from pathlib import Path

import torch
from torch import distributed as dist


def get_repo_path():
    curr_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return curr_dir.parent


def get_job_directory(file_or_checkpoint: Union[str, Dict[str, Any]]) -> str:
    found = False
    if isinstance(file_or_checkpoint, dict):
        chkpnt = file_or_checkpoint
        key = [x for x in chkpnt["callbacks"].keys() if "Checkpoint" in x][0]
        file = chkpnt["callbacks"][key]["dirpath"]
    else:
        file = file_or_checkpoint

    hydra_files = []
    directory = os.path.dirname(file)
    while not found:
        hydra_files = glob(
            os.path.join(os.path.join(directory, ".hydra/config.yaml")),
            recursive=True,
        )
        if len(hydra_files) > 0:
            break
        directory = os.path.dirname(directory)
        if directory == "":
            raise ValueError("Failed to find hydra config!")
    assert len(hydra_files) == 1, "Found ambiguous hydra config files!"
    job_dir = os.path.dirname(os.path.dirname(hydra_files[0]))
    return job_dir


def restore_model(checkpoint, pl_name="gsbm.pl_model", device=None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    job_dir = get_job_directory(checkpoint)
    cfg = OmegaConf.load(os.path.join(job_dir, ".hydra/config.yaml"))
    # print(f"Loaded cfg from {job_dir=}!")

    from .dataset import get_dist_boundary

    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    pl_module = importlib.import_module(f"{pl_name}")
    model = pl_module.GSBMLitModule(cfg, p0, p1, p0_val, p1_val)
    model.load_state_dict(ckpt["state_dict"])

    if device is not None:
        model = model.to(device)

    return model, cfg


def chunk_multi_output(input, chunk_op, split_size):
    """
    input: (B, *)
    chunk_op: (b, *) --> [(b, *), (b, *), ...]
    ===
    output: [(B, *), (B, *), ...]
    """
    B = input.shape[0]
    input_chunks = torch.split(input, split_size)

    output = None
    for chunk_idx, input_chunk in enumerate(input_chunks):
        output_chunk = chunk_op(input_chunk, chunk_idx)
        assert isinstance(output_chunk, tuple)

        # initialize output format
        if output is None:
            output = [[] for _ in range(len(output_chunk))]

        for n, o in enumerate(output_chunk):
            output[n].append(o)

    for n in range(len(output)):
        output[n] = torch.cat(output[n], dim=0)
        assert output[n].shape[0] == B
    return output


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def all_gather(tensor: torch.Tensor):
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(gathered_tensor, tensor)
    gathered_tensor = torch.cat(gathered_tensor, dim=0)
    return gathered_tensor


def gather(outputs, key):
    return torch.cat([o[key].detach().cpu() for o in outputs], dim=0)


def n_device():
    return dist.get_world_size() if is_distributed() else 1
