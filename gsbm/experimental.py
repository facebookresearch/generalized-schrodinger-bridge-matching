"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math

import torch
from torchdiffeq import odeint

from .sde import DIRECTIONS
from .gaussian_path import EndPointGaussianPath

################################################################################################
############################### Brownian bridge + quadratic cost ###############################
################################################################################################


# marginal sample
def qt_quad_cost(t, x0, x1, sigma, alpha):

    B, D = x0.shape
    assert x1.shape == (B, D) and t.shape == (B,)

    eta = torch.tensor(sigma * math.sqrt(2 * alpha))
    bar_eta = eta * (1 - t)

    ct = torch.sinh(bar_eta) / torch.sinh(eta)
    et = torch.sinh(bar_eta) * (1.0 / torch.tanh(bar_eta) - 1.0 / torch.tanh(eta))

    mean_t = ct[..., None] * x0 + et[..., None] * x1  # (B, D)
    cov_t = sigma**2 * et * torch.sinh(bar_eta) / eta
    std_t = (cov_t).sqrt()[..., None]  # (B,1)
    xt = mean_t + std_t * torch.randn_like(mean_t)

    assert xt.shape == x0.shape == x1.shape
    return xt


# optimal drift
def ut_quad_cost(t, x0, x1, xt, direction, sigma, alpha):
    assert direction in DIRECTIONS

    eta = torch.tensor(sigma * math.sqrt(2 * alpha))

    if direction == "fwd":
        c1 = eta / torch.sinh(eta * (1 - t))
        c2 = eta / torch.tanh(eta * (1 - t))
        ut = c1[..., None] * x1 - c2[..., None] * xt
    else:
        c1 = eta / torch.sinh(eta * t)
        c2 = eta / torch.tanh(eta * t)
        ut = c1[..., None] * x0 - c2[..., None] * xt
    assert ut.shape == x0.shape == x1.shape
    return ut


################################################################################################
############################### Simfree Gaussain path trajectory ###############################
################################################################################################


class EndPointGaussianPathv2(EndPointGaussianPath):
    odeint_kwargs = {
        "method": "scipy_solver",
        "atol": 1e-4,
        "rtol": 1e-7,
    }
    cov_eps = 1e-7
    cov_decom = "cholesky"

    def cov_mtx(self, t):
        """
        t: (T, ) --> cov_mtx: (B, T, T)
        """
        B, (T,) = self.B, t.shape

        Var_t = self.gamma(t).squeeze(-1) ** 2
        gt = odeint_gt(B, t, self.gamma, self.sigma, "cpu", self.odeint_kwargs)
        assert Var_t.shape == gt.shape == (B, T)

        Tidx, Sidx = torch.meshgrid(torch.arange(T), torch.arange(T))
        min_st = torch.min(Tidx, Sidx)
        max_st = torch.max(Tidx, Sidx)
        C = (gt[:, max_st] - gt[:, min_st]).exp() * Var_t[:, min_st]
        assert C.shape == (B, T, T)
        return C + self.cov_eps

    def sample_xs(self, T, N, eps=0.01):
        """joint
        t: (T)
        xs: (B, N, T, D)
        """
        B, D, device = self.B, self.D, self.device

        t = torch.linspace(eps, 1 - eps, T, device=device)

        ## Compute covariance & its decomposition
        C = self.cov_mtx(t)
        if self.cov_decom == "cholesky":
            A = torch.linalg.cholesky(C)
        elif self.cov_decom == "eigh":
            L, Q = torch.linalg.eigh(C)
            A = Q @ torch.diag_embed(L).sqrt()
        assert C.shape == A.shape == (B, T, T)

        ## Compute std_t and mean_t
        noise_t = torch.randn(N, B, T, D, device=device)
        std_t = (A @ noise_t).transpose(0, 1)  # <-- this took most memory
        assert std_t.shape == (B, N, T, D)

        torch.cuda.empty_cache()  # clear out memory

        mean_t = self.mean(t)  # (B, T, D)
        assert mean_t.shape == (B, T, D)

        ## sim-free xs
        xs = mean_t.unsqueeze(1) + std_t
        assert xs.shape == (B, N, T, D)
        return t, xs


def odeint_gt(B, ts, gamma, sigma, device, ocfg):
    """Implementation of Eq 32
    ts: (T,)
    gamma: (S,) -> (B, S)
    ===
    gt: (B, T)
    """
    (T,) = ts.shape
    orig_device = gamma.device

    def f(t, _gt):
        t = t.reshape(1)
        std, dstd = torch.autograd.functional.jvp(
            gamma, t, torch.ones_like(t), create_graph=False
        )
        B, T, D = std.shape
        assert dstd.shape == (B, T, D) and _gt.shape == (B, T)
        return (dstd - sigma**2 / (2 * std)) / std

    ts = ts.to(device)
    gamma = gamma.to(device)
    g0 = torch.zeros(B, 1, device=device)
    gt = odeint(f, g0, ts, **ocfg)
    gt = gt.transpose(0, 1).squeeze(-1)
    assert gt.shape == (B, T)

    gamma = gamma.to(orig_device)
    return gt.to(orig_device)
