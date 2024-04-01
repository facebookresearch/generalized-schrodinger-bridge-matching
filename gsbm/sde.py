"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import numpy as np
import torch
from tqdm import trange
from ipdb import set_trace as debug

DIRECTIONS = ["fwd", "bwd"]


def build_basedrift(cfg):
    if cfg.prob.name == "opinion":
        from .opinion import PolarizeDyn

        basedrift = PolarizeDyn(cfg.pdrift)
    else:
        basedrift = ZeroBaseDrift()
    return basedrift


class ZeroBaseDrift(torch.nn.Module):
    def __init__(self):
        super(ZeroBaseDrift, self).__init__()

    def forward(self, xt, t):
        return torch.zeros_like(xt)


def _assert_increasing(ts: torch.Tensor) -> None:
    assert (ts[1:] > ts[:-1]).all(), "time must be strictly increasing"


def sdeint(
    xinit,
    drift,
    diffusion,
    direction,
    nfe,
    log_steps=5,
    eps=0,
    verbose=False,
    return_ws=False,
):
    """
    xinit: (B, D)
    drift:     (B, D) + (B,) --> (B, D)
    diffusion: (B, D) + (B,) --> (B, D)
    nfe = T - 1
    ===
    t: (S,)
    xs: (B, S, D)
    us: (B, S, D)
    """
    assert direction in DIRECTIONS

    T, (B, D), S, device = nfe + 1, xinit.shape, log_steps, xinit.device

    # build ts
    timesteps = (
        torch.linspace(eps, 1 - eps, T, device=device)
        if direction == "fwd"
        else torch.linspace(1 - eps, eps, T, device=device)
    )

    # logging
    steps_to_log = np.linspace(0, T - 1, log_steps).astype(int)
    xs = []
    us = []
    if return_ws:
        ws = []

    x = xinit.detach()
    bar = trange(nfe) if verbose else range(nfe)
    for idx in bar:
        t, tnxt = timesteps[idx], timesteps[idx + 1]
        dt = (tnxt - t).abs()
        dw = math.sqrt(dt) * torch.randn(*x.shape, device=x.device)

        tt = t.repeat(x.shape[0])
        u = drift(x, tt)
        g = diffusion(x, tt)

        if idx in steps_to_log:
            xs.append(x)
            us.append(u)
            if return_ws:
                ws.append(dw)

        x = x + u * dt + g * dw

    assert len(xs) == len(us) == S - 1

    # log last step, u = 0
    xs.append(x)
    us.append(torch.zeros_like(x))
    if return_ws:
        ws.append(torch.zeros_like(x))
    ts = timesteps[steps_to_log]

    if direction == "bwd":
        xs = xs[::-1]
        us = us[::-1]
        ts = ts.flip(
            dims=[
                0,
            ]
        )
        if return_ws:
            ws = ws[::-1]

    xs = torch.stack(xs, dim=1).detach()
    us = torch.stack(us, dim=1).detach()
    assert xs.shape == us.shape == (B, S, D) and ts.shape == (S,)
    _assert_increasing(ts)
    out = {"t": ts, "xs": xs, "us": us}
    if return_ws:
        ws = torch.stack(ws, dim=1).detach()
        assert ws.shape == us.shape
        out["ws"] = ws
    return out
