"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import torch.nn.functional as F


def linear_interp1d(t, xt, mask, s):
    """Linear splines.
    B: batch, T: timestep, D: dim, S: query timestep
    Inputs:
        t: (T, B)
        xt: (T, B, D)
        mask: (T, B)
        s: (S, B)
    Outputs:
        xs: (S, B, D)
    """
    T, N, D = xt.shape
    S = s.shape[0]

    if mask is None:
        mask = torch.ones_like(t).bool()

    m = (xt[1:] - xt[:-1]) / (t[1:] - t[:-1] + 1e-10).unsqueeze(-1)

    left = torch.searchsorted(t[1:].T.contiguous(), s.T.contiguous(), side="left").T
    mask_l = F.one_hot(left, T).permute(0, 2, 1).reshape(S, T, N, 1)

    t = t.reshape(1, T, N, 1)
    xt = xt.reshape(1, T, N, D)
    m = m.reshape(1, T - 1, N, D)
    s = s.reshape(S, N, 1)

    x0 = torch.sum(t * mask_l, dim=1)
    p0 = torch.sum(xt * mask_l, dim=1)
    m0 = torch.sum(m * mask_l[:, :-1], dim=1)

    t = s - x0

    return t * m0 + p0


def cubic_interp1d(t, xt, mask, s):
    """
    Inputs:
        t: (T, N)
        xt: (T, N, D)
        mask: (T, N)
        s: (S, N)
    """
    T, N, D = xt.shape
    S = s.shape[0]

    if t.shape == s.shape:
        if torch.linalg.norm(t - s) == 0:
            return xt

    if mask is None:
        mask = torch.ones_like(t).bool()

    mask = mask.unsqueeze(-1)

    fd = (xt[1:] - xt[:-1]) / (t[1:] - t[:-1] + 1e-10).unsqueeze(-1)
    # Set tangents for the interior points.
    m = torch.cat([(fd[1:] + fd[:-1]) / 2, torch.zeros_like(fd[0:1])], dim=0)
    # Set tangent for the right end point.
    m = torch.where(torch.cat([mask[2:], torch.zeros_like(mask[0:1])]), m, fd)
    # Set tangent for the left end point.
    m = torch.cat([fd[[0]], m], dim=0)

    mask = mask.squeeze(-1)

    left = torch.searchsorted(t[1:].T.contiguous(), s.T.contiguous(), side="left").T
    right = (left + 1) % mask.sum(0).long()
    mask_l = F.one_hot(left, T).permute(0, 2, 1).reshape(S, T, N, 1)
    mask_r = F.one_hot(right, T).permute(0, 2, 1).reshape(S, T, N, 1)

    t = t.reshape(1, T, N, 1)
    xt = xt.reshape(1, T, N, D)
    m = m.reshape(1, T, N, D)
    s = s.reshape(S, N, 1)

    x0 = torch.sum(t * mask_l, dim=1)
    x1 = torch.sum(t * mask_r, dim=1)
    p0 = torch.sum(xt * mask_l, dim=1)
    p1 = torch.sum(xt * mask_r, dim=1)
    m0 = torch.sum(m * mask_l, dim=1)
    m1 = torch.sum(m * mask_r, dim=1)

    dx = x1 - x0
    t = (s - x0) / (dx + 1e-10)

    return (
        t**3 * (2 * p0 + m0 - 2 * p1 + m1)
        + t**2 * (-3 * p0 + 3 * p1 - 2 * m0 - m1)
        + t * m0
        + p0
    )
