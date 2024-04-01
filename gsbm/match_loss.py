"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import copy

import numpy as np
import torch
from torch.func import vmap, grad, jacrev

from .sde import DIRECTIONS

from ipdb import set_trace as debug


def laplacian(s):
    """Accepts a function s:R^D -> R."""
    H = jacrev(grad(s))
    return lambda x, t: torch.trace(H(x, t))


# entropic action matching loss + analytic laplacian
def _eam_inter_exact(net, xt, t, sigma):

    st = net(xt, t)
    dsdx, dsdt = torch.autograd.grad(st.sum(), (xt, t), create_graph=True)

    control = 0.5 * (dsdx**2).sum(dim=1, keepdim=True)
    dsdt = dsdt.reshape(-1, 1)

    lap = vmap(laplacian(lambda x, t: net(x, t).sum()), in_dims=(0, 0))(xt, t)
    ent_term = sigma**2 / 2 * lap.reshape(-1, 1)

    return control, dsdt, ent_term


def _eam_inter_approx(net, xt, t, sigma):

    dsdt_fn = grad(lambda x, t: net(x, t).sum(), argnums=1)
    dsdx_fn = grad(lambda x, t: net(x, t).sum(), argnums=0)

    eps = torch.randint(low=0, high=2, size=xt.shape).to(xt.device).float() * 2 - 1
    dsdx, jvp_val = torch.autograd.functional.jvp(
        lambda x: dsdx_fn(x, t), (xt,), (eps,), create_graph=True
    )
    lap = (jvp_val * eps).sum(1, keepdims=True)

    dsdt = dsdt_fn(xt, t)

    control = 0.5 * (dsdx**2).sum(dim=1, keepdim=True)
    dsdt = dsdt.reshape(-1, 1)
    ent_term = 0.5 * sigma**2 * lap

    return control, dsdt, ent_term


# entropic action matching loss with additional traj dimension
def eam_loss_trajs(net, xs, t, x0, x1, sigma, direction, lap="approx"):
    assert direction in DIRECTIONS

    B, T, D = xs.shape
    assert x0.shape == x1.shape == (B, D)
    assert t.shape == (T,)

    # boundary terms
    s0 = net(x0, torch.zeros(B, device=x0.device))
    s1 = net(x1, torch.ones(B, device=x1.device))

    # intermidiate terms
    xs.requires_grad_(True)
    t.requires_grad_(True)

    xs_N = xs.reshape(-1, D)
    t_N = t.reshape(1, T).expand(B, -1).reshape(-1)

    if lap == "approx":
        control, dsdt, ent_term = _eam_inter_approx(net, xs_N, t_N, sigma)
    elif lap == "exact":
        control, dsdt, ent_term = _eam_inter_exact(net, xs_N, t_N, sigma)
    else:
        raise ValueError(f"Unsupported analytic qt: {lap}!")

    control = control.reshape(B, T)
    dsdt = dsdt.reshape(B, T)
    ent_term = ent_term.reshape(B, T)

    ### reweight loss
    # if direction == "fwd":
    #     w_t_fn = lambda t: t
    #     dwdt_fn = lambda t: 1
    #     return (w_t_fn(0) * s0 - w_t_fn(1) * s1 + w_t_fn(t) * inter + dwdt_fn(t) * net(xt, t)).mean()
    # else:
    #     w_t_fn = lambda t: -t
    #     dwdt_fn = lambda t: -1
    #     return (-w_t_fn(0) * s0 + w_t_fn(1) * s1 + w_t_fn(t) * inter + dwdt_fn(t) * net(xt, t)).mean()

    if direction == "fwd":
        return (s0 - s1 + (control + dsdt + ent_term).mean(dim=1)).mean()
    elif direction == "bwd":
        return (-s0 + s1 + (control - dsdt + ent_term).mean(dim=1)).mean()
    else:
        raise ValueError(f"Unsupported direction option: {direction}!")


# bridge matching loss
def bm_loss(drift, xt, t, vt):
    pred_vt = drift(xt, t)
    assert pred_vt.shape == vt.shape == xt.shape
    return torch.square(pred_vt - vt).mean()
