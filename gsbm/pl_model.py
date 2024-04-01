"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, List
import os
import numpy as np
import math
from datetime import datetime
from rich.console import Console
from easydict import EasyDict as edict
import copy

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.utils as tu

from .network import build_net
from .state_cost import build_state_cost
from .evaluator import build_evaluator
from .sde import build_basedrift, sdeint

from .dataset import PairDataset, SplineDataset, SplineIWDataset
from . import gaussian_path as gpath_lib
from . import path_integral as pi_lib
from . import match_loss as match_lib
from . import utils

from .plotting import (
    save_fig,
    save_xs,
    plot_gpath,
    plot_iw,
    plot_boundaries,
    plot_xs_opinion,
)

# put gc.collect after io writing to prevent c10::CUDAError in multi-threading
# https://github.com/pytorch/pytorch/issues/67978#issuecomment-1661986812
import gc
from ipdb import set_trace as debug

console = Console()


class GSBMLitModule(pl.LightningModule):
    def __init__(self, cfg, p0, p1, p0_val, p1_val):
        super().__init__()

        os.makedirs("figs", exist_ok=True)

        self.cfg = cfg
        self.p0 = p0
        self.p1 = p1
        self.p0_val = p0_val
        self.p1_val = p1_val

        ### Problem
        self.sigma = cfg.prob.sigma
        self.V = build_state_cost(cfg)
        self.basedrift = build_basedrift(cfg)
        self.evaluator = build_evaluator(cfg)

        ### SB Model
        self.direction = None
        self.fwd_net = build_net(cfg)
        self.bwd_net = build_net(cfg)

    def print(self, content, prefix=True):
        if self.trainer.is_global_zero:
            if prefix:
                now = f"[[cyan]{datetime.now():%Y-%m-%d %H:%M:%S}[/cyan]]"
                if self.direction is None:
                    base = f"[[blue]Init[/blue]] "
                else:
                    base = f"[[blue]Ep {self.current_epoch} ({self.direction})[/blue]] "
                console.print(now, highlight=False, end="")
                console.print(base, end="")
            console.print(f"{content}")

    @property
    def wandb_logger(self):
        ## assume wandb is added to the end of loggers
        return self.loggers[-1]

    @property
    def is_img_prob(self):
        return self.cfg.prob.name in [
            "afhq",
        ]

    @property
    def logging_batch_idxs(self):
        return np.linspace(0, self.trainer.num_training_batches - 1, 10).astype(int)

    @property
    def ocfg(self):
        return self.cfg.optim

    @property
    def ccfg(self):
        return self.cfg.csoc

    @property
    def mcfg(self):
        return self.cfg.matching

    @property
    def pcfg(self):
        if self.cfg.prob.name == "lidar":
            pcfg = edict(self.cfg.plot)
            pcfg.dataset = self.V.dataset
            return pcfg
        return self.cfg.plot

    @property
    def device(self):
        return self.fwd_net.parameters().__next__().device

    @property
    def net(self):
        return self.fwd_net if self.direction == "fwd" else self.bwd_net

    @property
    def direction_r(self):
        return "bwd" if self.direction == "fwd" else "fwd"

    def build_ft(self, direction):
        def ft(x, t):
            """
            x: (B, D)
            t: (B,)
            ===
            out: (B, D)
            """
            B, D = x.shape
            sign = 1.0 if direction == "fwd" else -1.0
            assert t.shape == (B,) and torch.allclose(t, t[0] * torch.ones_like(t))
            return sign * self.basedrift(x.unsqueeze(1), t[0].reshape(1)).squeeze(1)

        return ft

    def build_ut(self, direction, backprop_snet=False):
        """
        ut: x: (B, D), t: (B,) --> (B, D)
        """
        net = self.fwd_net if direction == "fwd" else self.bwd_net
        if self.cfg.field == "vector":
            ut = net
        elif self.cfg.field == "potential":

            def ut(x, t):
                with torch.enable_grad():
                    x = x.detach().clone()
                    x.requires_grad_(True)
                    out = net(x, t)
                    return torch.autograd.grad(
                        out.sum(), x, create_graph=backprop_snet
                    )[0]

        else:
            ValueError(f"Unsupportted field: {self.cfg.field}!")
        return ut

    def build_drift(self, direction, backprop_snet=False):
        ft = self.build_ft(direction)
        ut = self.build_ut(direction, backprop_snet=backprop_snet)
        drift = lambda x, t: ut(x, t) + ft(x, t)
        return drift

    @torch.no_grad()
    def sample(self, xinit, log_steps, direction, drift=None, nfe=None, verbose=False):
        drift = self.build_drift(direction) if drift is None else drift
        diffusion = lambda x, t: self.sigma
        nfe = nfe or self.cfg.nfe
        output = sdeint(
            xinit,
            drift,
            diffusion,
            direction,
            nfe=nfe,
            log_steps=log_steps,
            verbose=verbose,
        )
        return output

    def sample_t(self, batch):
        if self.mcfg.loss == "eam":
            t0 = torch.rand(1)
            t = (t0 + math.sqrt(2) * torch.arange(batch)) % 1
            t.clamp_(min=0.001, max=0.999)
        elif self.mcfg.loss == "bm":
            eps = 1e-4
            t = torch.rand(batch).reshape(-1) * (1 - 2 * eps) + eps
        else:
            raise ValueError(f"Unsupported matching loss option: {self.mcfg.loss}!")

        assert t.shape == (batch,)
        return t

    def sample_gpath(self, batch):
        ### Setup
        gpath = gpath_lib.EndPointGaussianPath(
            batch["mean_t"][0],
            batch["mean_xt"],
            batch["gamma_s"][0],
            batch["gamma_xs"],
            self.sigma,
            self.basedrift,
        )
        x0, x1 = batch["x0"], batch["x1"]
        B, D = x0.shape

        ### Sample t and xt
        T = B if self.mcfg.loss == "bm" else self.mcfg.batch_t
        if not self.ccfg.IW:
            t = self.sample_t(T).to(x0)
            with torch.no_grad():
                xt = gpath.sample_xt(t, N=1)
        else:
            IW_t, IW_xs, weights = batch["IW_t"][0], batch["IW_xs"], batch["weights"]

            # weights should be positive and self-normalized
            assert (weights > 0).all()
            assert torch.allclose(weights.sum(dim=1), torch.ones(B).to(weights))

            rand_idx = torch.randint(low=0, high=len(IW_t), size=(T,))
            t = IW_t[rand_idx]
            xt = pi_lib.impt_weighted(t, IW_xs, weights).unsqueeze(1)
        assert t.shape == (T,) and xt.shape == (B, 1, T, D)

        ### Sample vt and build output
        if self.mcfg.loss == "bm":
            assert B == T
            vt = gpath.ut(t, xt, self.direction)
            xt = xt[torch.arange(B), 0, torch.arange(B)]
            vt = vt[torch.arange(B), 0, torch.arange(B)]
            assert xt.shape == vt.shape == (B, D)

        elif self.mcfg.loss == "eam":
            vt = None
            xt = xt.squeeze(1)
            assert xt.shape == (B, T, D)

        return x0, x1, t, xt, vt

    def training_step(self, batch: Any, batch_idx: int):
        ### Sample from Gaussian path
        x0, x1, t, xt, vt = self.sample_gpath(batch)

        ### Apply bridge matching orr entropic action matching
        if self.mcfg.loss == "bm":
            ut = self.build_ut(self.direction, backprop_snet=True)
            loss = match_lib.bm_loss(ut, xt, t, vt)
        elif self.mcfg.loss == "eam":
            loss = match_lib.eam_loss_trajs(
                self.net,
                xt,
                t,
                x0,
                x1,
                self.sigma,
                self.direction,
                lap=self.mcfg.lap,
            )
        else:
            raise ValueError(f"Unsupported match_loss option: {self.mcfg.loss}!")

        if torch.isfinite(loss):
            self.log("train/loss", loss, on_step=True, on_epoch=True)
        else:
            ### Skip step if loss is NaN.
            self.print(f"Skipping iteration because loss is {loss.item()}.")
            return None

        if batch_idx in self.logging_batch_idxs:
            self.print(
                f"[M-step] batch idx: {batch_idx+1}/{self.trainer.num_training_batches} ..."
            )

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        ### ** The only place where we modify the direction!! **
        self.direction = self.direction_r
        self.print("", prefix=False)  # change line

    def localize(self, p):
        g = torch.Generator()
        g.manual_seed(g.seed() + self.global_rank)
        local_p = copy.deepcopy(p)
        local_p.set_generator(g)
        return local_p

    def val_dataloader(self):
        totalB, n_device = self.ccfg.B, utils.n_device()
        B = totalB // n_device
        self.print(f"[Data] Building {totalB} train_data ...")
        self.print(
            f"[Data] Found {n_device} devices, each will generate {B} samples ..."
        )

        x0 = self.localize(self.p0)(B)
        x1 = self.localize(self.p1)(B)
        return DataLoader(
            PairDataset(x0, x1),
            num_workers=self.ocfg.num_workers,
            batch_size=self.ccfg.mB,
            persistent_workers=self.ocfg.num_workers > 0,
            shuffle=False,
            pin_memory=True,
        )

    def compute_coupling(self, batch, direction, eval_coupling):
        x0, x1, T = batch["x0"], batch["x1"], self.ccfg.T_mean
        if direction is None:
            t = torch.linspace(0, 1, T).to(x0)
            xt = (1 - t[None, :, None]) * x0.unsqueeze(1) + t[
                None, :, None
            ] * x1.unsqueeze(1)
        else:
            xinit = x0 if direction == "fwd" else x1
            output = self.sample(xinit, log_steps=T, direction=direction)
            t, xt = output["t"], output["xs"]

            if eval_coupling:
                metrics = self.evaluator(output)
                for k, v in metrics.items():
                    self.log(f"metrics/{k}", v, on_epoch=True)

        return t, xt

    def validation_step(self, batch: Any, batch_idx: int):
        log_step = batch_idx == 0
        ccfg, direction = self.ccfg, self.direction
        postfix = f"{self.current_epoch:03d}" if direction is not None else "init"

        (B, D), T, S, sigma = batch["x0"].shape, ccfg.T_mean, ccfg.T_gamma, self.sigma

        ### Initialize mean spline (with copuling)
        eval_coupling = log_step and self.cfg.eval_coupling
        self.print(f"[R-step] Simulating {direction or 'init'} coupling ...")
        t, xt = self.compute_coupling(batch, direction, eval_coupling)
        self.print(f"[R-step] Simulated {xt.shape=}!")
        if log_step:
            self.log_coupling(t, xt, direction, f"coupling-{postfix}")
        assert xt.shape == (B, T, D) and t.shape == (T,)

        ### Initialize std spline
        s = torch.linspace(0, 1, S).to(t)
        ys = torch.zeros(B, S, 1).to(xt)

        ### Fit Gaussian paths (update xt, ys)
        gpath = gpath_lib.EndPointGaussianPath(t, xt, s, ys, sigma, self.basedrift)
        if self.is_img_prob:
            loss_fn = gpath_lib.build_img_loss_fn(gpath, sigma, self.V, ccfg)
        else:
            loss_fn = gpath_lib.build_loss_fn(gpath, sigma, self.V, ccfg)
        with torch.enable_grad():
            verbose = log_step and self.trainer.is_global_zero
            result = gpath_lib.fit(
                ccfg, gpath, direction or "fwd", loss_fn, verbose=verbose
            )
        self.print(f"[R-step] Fit {B} gaussian paths!")
        if log_step:
            self.log_gpath(result, f"gpath-{postfix}")

        ### Built output
        xt = gpath.mean.xt.detach().clone()
        ys = gpath.gamma.xt.detach().clone()
        assert xt.shape == (B, T, D) and ys.shape == (B, S, 1)
        output = {"mean_t": t, "mean_xt": xt, "gamma_s": s, "gamma_xs": ys}

        ### (optional) Handle important weighting
        if ccfg.IW:
            with torch.no_grad():
                iw_output = pi_lib.impt_sample_xs(
                    ccfg, gpath, sigma, direction or "fwd", V=self.V
                )
            output.update(iw_output)
            self.print(f"[R-step] Compute IW.shape={iw_output['weights'].shape}!")
            if log_step:
                self.log_IW(iw_output, f"iw-{postfix}")

        ### (optional) Handle opinion drift
        if ccfg.name == "opinion":
            tt = torch.linspace(0, 1, self.cfg.pdrift.S).to(t)
            mf_x = gpath.sample_xt(tt, N=1).squeeze(1)
            assert mf_x.shape == (B, len(tt), D)
            output["mf_x"] = mf_x.detach().cpu()

        return output

    def validation_epoch_end(self, outputs: List[Any]):
        ### Handle opinion drift
        if self.cfg.prob.name == "opinion":
            mf_xs = utils.gather(outputs, "mf_x")
            # if utils.is_distributed():
            #     mf_xs = utils.all_gather(mf_xs)
            self.basedrift.set_mf_drift(mf_xs)
            self.print(f"[Opinion] Set MF drift shape={mf_xs.shape}!")

        ccfg = self.ccfg
        T, S, D = ccfg.T_mean, ccfg.T_gamma, self.cfg.dim

        ## gather mean_t, gamma_s
        mean_t = outputs[0]["mean_t"].detach().cpu()
        gamma_s = outputs[0]["gamma_s"].detach().cpu()
        assert mean_t.shape == (T,) and gamma_s.shape == (S,)

        ## gather mean_xt, gamma_xs
        mean_xt = utils.gather(outputs, "mean_xt")
        gamma_xs = utils.gather(outputs, "gamma_xs")
        B = mean_xt.shape[0]
        assert mean_xt.shape == (B, T, D)
        assert gamma_xs.shape == (B, S, 1)

        self.train_data = SplineDataset(
            mean_t, mean_xt, gamma_s, gamma_xs, expand_factor=ccfg.epd_fct
        )
        self.print(f"[Data] Fit total {B} gaussian paths as train_data!")

        ### (optional) IW
        if ccfg.IW:
            iN, iS = ccfg.IW_N, ccfg.IW_S
            IW_t = outputs[0]["IW_t"].detach().cpu()
            IW_xs = utils.gather(outputs, "IW_xs")
            weights = utils.gather(outputs, "weights")
            assert IW_t.shape == (iS,) and IW_xs.shape == (B, iN, iS, D)
            assert weights.shape == (B, iN)
            self.print(f"[Data] Computed important {weights.shape=} as train_data!")
            self.train_data = SplineIWDataset(self.train_data, IW_t, IW_xs, weights)

        if self.direction is None:
            self.direction = "fwd"
            self.print("", prefix=False)  # change line

        torch.cuda.empty_cache()

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_data,
            num_workers=self.ocfg.num_workers,
            batch_size=self.ocfg.batch_size,
            persistent_workers=self.ocfg.num_workers > 0,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.ocfg.lr,
            weight_decay=self.ocfg.wd,
            eps=self.ocfg.eps,
        )

        if self.ocfg.get("scheduler", "cosine") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.ocfg.num_iterations,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.net.update_ema()

    def log_coupling(self, t, xs, direction, fn, log_steps=5):
        """
        t: (T,) xs: (B, T, D)
        """
        B, T, D = xs.shape
        assert t.shape == (T,)

        if self.is_img_prob:
            # subsample B & T to 10 & S
            mB = 10  # mini-batch
            xs = xs[:mB][:, np.linspace(0, T - 1, log_steps).astype(int)]
            xs = xs.reshape(mB * log_steps, D)
            self.log_images(xs, log_steps, fn, "viz")
        else:
            save_xs(t, xs, log_steps, direction, self.pcfg, fn)

        gc.collect()

    def log_gpath(self, result, fn):
        plot_gpath(result, self.pcfg)
        save_fig(fn)
        gc.collect()

        if self.is_img_prob:
            B, T, D = result["init_mean"].shape
            assert T < 10

            self.log_images(result["init_mean"][:10], T, f"{fn}-init_mean", "init_mean")
            self.log_images(
                result["final_mean"][:10], T, f"{fn}-final_mean", "final_mean"
            )

    def log_IW(self, result, fn):
        IW_t, IW_xs, ws = result["IW_t"], result["IW_xs"], result["weights"]
        result["IW_xt"] = pi_lib.impt_weighted(IW_t, IW_xs, ws)
        plot_iw(result, self.pcfg)
        save_fig(fn)
        gc.collect()

    def log_images(self, x, T, fn, key=None):
        """
        x: (B, 3*D*D) --> grid images: (3, (B//T)*D, T*D)
        """
        images = x.reshape(-1, *self.cfg.image_size)
        images = torch.clamp((images + 1) / 2, 0.0, 1.0).cpu()
        tu.save_image(images, f"figs/{fn}.png", nrow=T, pad_value=1.0)

        if key is not None and self.cfg.use_wandb:
            grid_image = tu.make_grid(images, nrow=T, pad_value=1.0)
            self.wandb_logger.log_image(key=f"images/{key}", images=[grid_image])

        gc.collect()

    def log_boundary(self, p0, p1, p0_val, p1_val):
        if self.is_img_prob:
            self.log_images(p0(64), 8, "init-p0")
            self.log_images(p1(64), 8, "init-p1")
            self.log_images(p0_val(64), 8, "init-p0-val")
            self.log_images(p1_val(64), 8, "init-p1-val")
        else:
            plot_boundaries(p0, p1, self.pcfg)
            save_fig("train_dist")
            plot_boundaries(p0_val, p1_val, self.pcfg)
            save_fig("val_dist")
            gc.collect()

    def log_basedrift(self, p0):
        ft = self.build_ft("fwd")
        result = self.sample(p0(512), log_steps=5, direction="fwd", drift=ft, nfe=500)
        plot_xs_opinion(result["t"], result["xs"], 5, "Init", self.pcfg)
        save_fig("ft")
