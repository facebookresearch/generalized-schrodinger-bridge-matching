"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import torch.nn as nn

from .ema import EMA
from .nn import (
    timestep_embedding,
    Unbatch,
    SiLU,
    ResNet_FC,
)
from ipdb import set_trace as debug


def build_net(cfg):
    if hasattr(cfg, "unet"):
        field = UNetVectorField(cfg.unet)
    else:
        field = {
            "toy-potential": ToyPotentialField,
            "toy-vector": ToyVectorField,
            "opinion-vector": OpinionVectorField,
        }.get(f"{cfg.net}-{cfg.field}")(cfg.dim)
    return EMA(Unbatch(field), cfg.optim.ema_decay)


class ToyPotentialField(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128):
        super(ToyPotentialField, self).__init__()

        self.xt_module = ResNet_FC(data_dim + 1, hidden_dim, num_res_blocks=3)

        self.out_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (b,nx)
        t: (b,)
        """
        h = torch.hstack([t.reshape(-1, 1), x])
        h = self.xt_module(h)
        out = self.out_module(h)
        return out


class ToyVectorField(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 128,
        step_scale: int = 1000,
    ):
        super(ToyVectorField, self).__init__()

        self.step_scale = step_scale
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = nn.Sequential(
            nn.Linear(data_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (b,nx)
        t: (b,)
        """

        steps = t * self.step_scale
        t_emb = timestep_embedding(steps, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out + t_out)

        return out


class OpinionVectorField(nn.Module):
    def __init__(
        self, data_dim=1000, hidden_dim=256, time_embed_dim=128, step_scale=1000
    ):
        super(OpinionVectorField, self).__init__()

        self.step_scale = step_scale
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(data_dim, hid, num_res_blocks=5)

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        """

        t = t * self.step_scale
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out + t_out)

        return out


class UNetVectorField(nn.Module):
    def __init__(self, cfg, timesteps=1000):
        super(UNetVectorField, self).__init__()

        from .unet import UNetModel

        self.net = UNetModel(**cfg)
        self.timesteps = timesteps

    def forward(self, x, t) -> torch.Tensor:
        """
        x: (b,nx) range: [-1,1]
        t: (b,)   timesteps
        """
        B, D = x.shape
        assert t.shape == (B,)
        assert D == 3 * 64 * 64

        batch = {}
        batch["noisy_x"] = x.reshape(B, 3, 64, 64)
        timestep = t * self.timesteps

        return self.net(batch, timestep).reshape(B, D)
