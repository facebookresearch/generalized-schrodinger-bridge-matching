"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch

from .sde import DIRECTIONS, sdeint

from ipdb import set_trace as debug


def t_to_idx(t: torch.Tensor, T: int) -> torch.Tensor:
    return (t * (T - 1)).round().long()


def impt_weight_fn(s, xs, us, ws, V, sigma):
    """
    s: (S,)
    xs: (B, N, S, D)
    us: (B, N, S, D)
    ws: (B, N, S, D)
    ===
    weights: (B, N)
    """

    B, N, S, D = xs.shape
    assert s.shape == (S,)
    assert (s < 1).all() and (s > 0).all()
    assert us.shape == ws.shape == (B, N, S, D)
    assert sigma > 0

    dt = torch.cat([s[[1]] - s[[0]], s[1:] - s[:-1]]).reshape(1, 1, -1)
    assert dt.shape == (1, 1, S) and (dt > 0).all()

    state_cost = V(xs, s) * dt
    control_cost = (0.5 / sigma**2) * (us**2).sum(dim=-1) * dt
    girsanov_cost = (1.0 / sigma) * (us * ws).sum(dim=-1)
    assert state_cost.shape == control_cost.shape == girsanov_cost.shape == (B, N, S)

    total_cost = (state_cost + control_cost + girsanov_cost).mean(dim=-1)  # (B, N)
    total_cost = total_cost - total_cost.min(dim=-1, keepdim=True)[0]  # (B, 1)
    assert total_cost.shape == (B, N)
    # print(state_cost.abs().max(), control_cost.abs().max(), girsanov_cost.abs().max())

    weights = torch.exp(-total_cost)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    assert weights.shape == (B, N)
    return weights


def impt_sample_xs(ccfg, gpath, sigma, direction, V, eps=0.001):
    assert direction in DIRECTIONS and sigma > 0

    B, D, N, S = gpath.B, gpath.D, ccfg.IW_N, ccfg.IW_S
    device = gpath.device

    tinit = eps if direction == "fwd" else (1.0 - eps)
    xinit = gpath.sample_xt(tinit * torch.ones(1, device=device), N=N)  # (B, N, 1, D)
    xinit = xinit.reshape(B * N, D)

    def drift(xt, t):
        """
        xt: (B*N, D)
        t: (B*N,)
        ===
        ut: (B*N, D)
        """
        assert torch.allclose(t, t[0] * torch.ones_like(t))
        _t = t[0].reshape(1)
        _xt = xt.reshape(B, N, 1, D)
        ut = gpath.ut(_t, _xt, direction)  # drift - ft
        ft = gpath.ft(_t, _xt, direction)
        assert _xt.shape == ut.shape == ft.shape

        return (ut + ft).reshape(B * N, D)

    diffusion = lambda x, t: sigma
    out = sdeint(
        xinit,
        drift,
        diffusion,
        direction,
        nfe=S - 1,
        log_steps=S,
        eps=eps,
        return_ws=True,
    )
    s, xs, us, ws = out["t"], out["xs"], out["us"], out["ws"]

    xs = xs.reshape(B, N, S, D)
    us = us.reshape(B, N, S, D)
    ws = ws.reshape(B, N, S, D)

    VV = lambda x, t: V(x, t, gpath)
    weights = impt_weight_fn(s, xs, us, ws, VV, sigma)
    return {"IW_t": s, "IW_xs": xs, "weights": weights}


def impt_weighted(t, xs, weights):
    """
    t: (T,)
    xs: (B, N, S, D)
    weights: (B, N)
    ===
    out: (B, T, D)
    """
    (T,), (B, N, S, D) = t.shape, xs.shape
    assert weights.shape == (B, N)

    permvector = torch.multinomial(weights, T, replacement=True)
    permvector = permvector.unsqueeze(-1).expand(-1, -1, D)
    assert permvector.shape == (B, T, D)

    ## subsample time grid
    ys = xs[:, :, t_to_idx(t, S)]
    assert ys.shape == (B, N, T, D)

    # https://github.com/pytorch/pytorch/issues/30574#issuecomment-1199665661
    yt = ys.gather(1, permvector.unsqueeze(1)).squeeze(1)
    assert yt.shape == (B, T, D)
    return yt
