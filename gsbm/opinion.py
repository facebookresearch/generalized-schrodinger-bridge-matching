"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import numpy as np
import torch

from ipdb import set_trace as debug


def t_to_idx(t: torch.Tensor, T: int) -> torch.Tensor:
    return (t * (T - 1)).round().long()


@torch.no_grad()
def est_directional_similarity(xs: torch.Tensor, n_est: int = 1000) -> torch.Tensor:
    """xs: (batch, nx). Returns (n_est, ) between 0 and 1."""
    # xs: (batch, nx)
    batch, nx = xs.shape

    # Center first.
    xs = xs - torch.mean(xs, dim=0, keepdim=True)

    rand_idxs1 = torch.randint(batch, [n_est], dtype=torch.long)
    rand_idxs2 = torch.randint(batch, [n_est], dtype=torch.long)

    # (n_est, nx)
    xs1 = xs[rand_idxs1]
    # (n_est, nx)
    xs2 = xs[rand_idxs2]

    # Normalize to unit vector.
    xs1 /= torch.linalg.norm(xs1, dim=1, keepdim=True)
    xs2 /= torch.linalg.norm(xs2, dim=1, keepdim=True)

    # (n_est, )
    cos_angle = torch.sum(xs1 * xs2, dim=1).clip(-1.0, 1.0)
    assert cos_angle.shape == (n_est,)

    # Should be in [0, pi).
    angle = torch.acos(cos_angle)
    assert (0 <= angle).all()
    assert (angle <= torch.pi).all()

    D_ij = 1.0 - angle / torch.pi
    assert D_ij.shape == (n_est,)

    return D_ij


def opinion_thresh(inner: torch.Tensor) -> torch.Tensor:
    return 2.0 * (inner > 0) - 1.0


def compute_mean_drift_term(mf_x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Decompose the polarize dynamic Eq (18) in paper into 2 parts for faster computation:
          f_polarize(x,p,ξ)
        = E_{y~p}[a(x,y,ξ) * bar_y],                  where a(x,y,ξ) = sign(<x,ξ>)*sign(<y,ξ>)
                                                      and      bar_y = y / |y|^{0.5}
        = sign(<x,ξ>) * E_{y~p}[sign(<y,ξ>) * bar_y], since sign(<x,ξ>) is independent of y
        = A(x,ξ)      * B(p,ξ)
    Hence, bar_f_polarize = bar_A(x,ξ) * bar_B(p,ξ)
    This function computes only bar_B(p,ξ).
    """
    # mf_x: (B, *, D), xi: (*, D)
    # output: (*, D)

    B, Ts, D = mf_x.shape[0], mf_x.shape[1:-1], mf_x.shape[-1]
    assert xi.shape == (*Ts, D)

    mf_x_norm = torch.linalg.norm(mf_x, dim=-1, keepdim=True)
    assert torch.all(mf_x_norm > 0.0)

    normalized_mf_x = mf_x / torch.sqrt(mf_x_norm)
    assert normalized_mf_x.shape == (B, *Ts, D)

    # Compute the mean drift term:   1/J sum_j a(y_j) y_j / sqrt(| y_j |).
    mf_agree_j = opinion_thresh(torch.sum(mf_x * xi, dim=-1, keepdim=True))
    assert mf_agree_j.shape == (B, *Ts, 1)

    mean_drift_term = torch.mean(mf_agree_j * normalized_mf_x, dim=0)
    assert mean_drift_term.shape == (*Ts, D)

    mean_drift_term_norm = torch.linalg.norm(mean_drift_term, dim=-1, keepdim=True)
    mean_drift_term = mean_drift_term / torch.sqrt(mean_drift_term_norm)
    assert mean_drift_term.shape == (*Ts, D)

    return mean_drift_term


def opinion_f(
    x: torch.Tensor, mf_drift: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    """This function computes the polarize dynamic in Eq (18) by
        bar_f_polarize(x,p,ξ) = bar_A(x,ξ) * bar_B(p,ξ)
    where bar_B(p,ξ) is pre-computed in func compute_mean_drift_term and passed in as mf_drift.
    """
    # x: (b, T, nx), mf_drift: (T, nx), xi: (T, nx)
    # out: (b, T, nx)

    b, T, nx = x.shape
    assert xi.shape == mf_drift.shape == (T, nx)

    agree_i = opinion_thresh(torch.sum(x * xi, dim=-1, keepdim=True))
    # Make sure we are not dividing by 0.
    agree_i[agree_i == 0] = 1.0

    abs_sqrt_agree_i = torch.sqrt(torch.abs(agree_i))
    assert torch.all(abs_sqrt_agree_i > 0.0)

    norm_agree_i = agree_i / abs_sqrt_agree_i
    assert norm_agree_i.shape == (b, T, 1)

    f = norm_agree_i * mf_drift
    assert f.shape == (b, T, nx)

    return f


def build_f_mul(T, coeff=8.0) -> torch.Tensor:
    # set f_mul with some heuristic so that it doesn't diverge exponentially fast
    # and yield bad normalization, since the more polarized the opinion is the faster it will grow
    ts = torch.linspace(0.0, 1.0, T)
    f_mul = torch.clip(1.0 - torch.exp(coeff * (ts - 1.0)) + 1e-5, min=1e-4, max=1.0)
    f_mul = f_mul**5.0
    return f_mul


def build_xis(T, D) -> torch.Tensor:
    # Generate random unit vectors.
    rng = np.random.default_rng(seed=4078213)
    xis = rng.standard_normal([T, D])

    # Construct a xis that has some degree of "continuous" over time, as a brownian motion.
    xi = xis[0]
    bm_xis = [xi]
    std = 0.4
    dt = 1.0 / T
    for t in range(1, T):
        xi = xi - (2.0 * xi) * dt + std * math.sqrt(dt) * xis[t]
        bm_xis.append(xi)
    assert len(bm_xis) == xis.shape[0]

    xis = torch.Tensor(np.stack(bm_xis))
    xis /= torch.linalg.norm(xis, dim=-1, keepdim=True)

    # Just safeguard if the self.xis becomes different.
    print("USING BM XI! xis.sum(): {}".format(torch.sum(xis)))
    assert xis.shape == (T, D)
    return xis


@torch.no_grad()
def proj_pca(xs_f: torch.Tensor):
    """
    xs_f: (B, T, D)
    ===
    proj_xs_f: (B, T, 2)
    V: (D, 2) s.t. proj_xs = xs @ V
    """
    # xs_f: (batch, T, nx)
    # Only use final timestep of xs_f for PCA.
    batch, T, nx = xs_f.shape

    # (batch * T, nx)
    flat_xsf = xs_f.reshape(-1, *xs_f.shape[2:])

    # Center by subtract mean.
    # (batch, nx)
    final_xs_f = xs_f[:, -1, :]

    mean_pca_xs = torch.mean(final_xs_f, dim=0, keepdim=True)
    final_xs_f -= mean_pca_xs

    # if batch is too large, it will run out of memory.
    if batch > 200:
        rand_idxs = torch.randperm(batch)[:200]
        final_xs_f = final_xs_f[rand_idxs]

    # U: (batch, k)
    # S: (k, k)
    # VT: (k, nx)
    U, S, VT = torch.linalg.svd(final_xs_f)

    # log.info("Singular values of xs_f at final timestep:")
    # log.info(S)

    # Keep the first and last directions.
    VT = VT[:2, :]
    # VT = VT[[0, -1], :]

    assert VT.shape == (2, nx)
    V = VT.T

    # Project both xs_f and xs_b onto V.
    flat_xsf -= mean_pca_xs

    proj_xs_f = flat_xsf @ V
    proj_xs_f = proj_xs_f.reshape(batch, T, *proj_xs_f.shape[1:])

    return proj_xs_f, V


class PolarizeDyn(torch.nn.Module):
    def __init__(self, pcfg) -> None:
        super(PolarizeDyn, self).__init__()

        self.S = pcfg.S
        self.D = pcfg.D
        self.polarize_strength = pcfg.strength
        self.register_buffer("xis", build_xis(pcfg.S, pcfg.D))
        self.register_buffer("f_muls", build_f_mul(pcfg.S, coeff=pcfg.m_coeff))
        self.register_buffer("mf_drift", torch.zeros(pcfg.S, pcfg.D))
        self.register_buffer("is_mf_drift_set", torch.tensor(False))

    def set_mf_drift(self, mf_xs):
        """run on cpu to prevent OOM
        mf_xs: (B, S, D)
        """
        assert mf_xs.shape[1:] == (self.S, self.D)

        xis = self.xis.detach().cpu()
        mf_drift = compute_mean_drift_term(mf_xs, xis)
        assert mf_drift.shape == (self.S, self.D)

        self.mf_drift = mf_drift.to(self.xis)
        self.is_mf_drift_set = torch.tensor(True).to(self.xis)

    def forward(self, xs, t):
        """
        xs: (*, T, D)
        t: (T,)
        ===
        out: (*, T, D)
        """
        Bs, (T, D) = xs.shape[:-2], xs.shape[-2:]
        assert t.shape == (T,)

        xs = xs.reshape(-1, T, D)

        t_idx = t_to_idx(t, self.S)
        fmul = self.f_muls[t_idx].to(xs)
        assert t_idx.shape == fmul.shape == (T,)

        xi = self.xis[t_idx].to(xs)
        if self.is_mf_drift_set:
            mf_drift = self.mf_drift[t_idx].to(xs)
        else:
            mf_drift = compute_mean_drift_term(xs, xi)
        assert xi.shape == mf_drift.shape == (T, D)

        f = self.polarize_strength * opinion_f(xs, mf_drift, xi)
        assert f.shape == xs.shape

        f = fmul.reshape(1, -1, 1) * f
        assert f.shape == xs.shape

        return f
