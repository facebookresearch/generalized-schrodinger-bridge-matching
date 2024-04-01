"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import os.path as osp
import sys
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import json
from glob import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from gsbm.dataset import get_dist_boundary
from gsbm.pl_model import GSBMLitModule

import colored_traceback.always

from ipdb import set_trace as debug

torch.backends.cudnn.benchmark = True
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.getLevelName("INFO"))

    hydra_config = HydraConfig.get()

    # Get the number of nodes we are training on
    nnodes = hydra_config.launcher.get("nodes", 1)
    print("nnodes", nnodes)

    if cfg.get("seed", None) is not None:
        pl.utilities.seed.seed_everything(cfg.seed)

    print(cfg)

    print("Found {} CUDA devices.".format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            "{} \t Memory: {:.2f}GB".format(props.name, props.total_memory / (1024**3))
        )

    keys = [
        "SLURM_NODELIST",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_JOB_NAME",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
    ]
    log.info(json.dumps({k: os.environ.get(k, None) for k in keys}, indent=4))

    cmd_str = " \\\n".join([f"python {sys.argv[0]}"] + ["\t" + x for x in sys.argv[1:]])
    with open("cmd.sh", "w") as fout:
        print("#!/bin/bash\n", file=fout)
        print(cmd_str, file=fout)

    log.info(f"CWD: {os.getcwd()}")

    # Construct model
    p0, p1, p0_val, p1_val = get_dist_boundary(cfg)
    model = GSBMLitModule(cfg, p0, p1, p0_val, p1_val)
    model.log_boundary(p0, p1, p0_val, p1_val)
    if cfg.prob.name == "opinion":
        model.log_basedrift(p0)
    # print(model)

    # Checkpointing, logging, and other misc.
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="epoch-{epoch:03d}_step-{step}",
            auto_insert_metric_name=False,
            save_top_k=-1,  # save all models whenever callback occurs
            save_last=True,
            every_n_epochs=1,
            verbose=True,
        ),
        LearningRateMonitor(),
    ]

    slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.use_wandb:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.prob.name}_{now}",
                project="GSBM",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )

    strategy = "ddp" if torch.cuda.device_count() > 1 else None

    trainer = pl.Trainer(
        max_epochs=cfg.optim.max_epochs,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        num_nodes=nnodes,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
        reload_dataloaders_every_n_epochs=1,  # GSBM: refresh on-policy samples every epoch
        num_sanity_val_steps=-1,  # GSBM: validate before training -> radom coupling
        check_val_every_n_epoch=1,  # GSBM: validate -> markovian coupling
        replace_sampler_ddp=False,  # GSBM: avoid gather_all, use device-wise dataloader
        enable_progress_bar=False,
    )

    # If we specified a checkpoint to resume from, use it
    checkpoint = cfg.get("resume", None)

    # Check if a checkpoint exists in this working directory.  If so, then we are resuming from a pre-emption
    # This takes precedence over a command line specified checkpoint
    checkpoints = glob("checkpoints/**/*.ckpt", recursive=True)
    if len(checkpoints) > 0:
        # Use the checkpoint with the latest modification time
        checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    # Load dataset (train loader will be generated online)
    trainer.fit(model, ckpt_path=checkpoint)

    metric_dict = trainer.callback_metrics

    for k, v in metric_dict.items():
        metric_dict[k] = float(v)

    with open("metrics.json", "w") as fout:
        print(json.dumps(metric_dict), file=fout)

    return metric_dict


if __name__ == "__main__":
    main()
