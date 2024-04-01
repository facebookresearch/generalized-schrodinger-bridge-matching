"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math
import copy

import numpy as np
import torch
from tqdm import trange

from . import interp1d
from .sde import DIRECTIONS

from ipdb import set_trace as debug

################################################################################################


class EndPointSpline(torch.nn.Module):
    def __init__(self, t, xt, spline_type="linear"):
        """
        t: (T,)
        xt: (B, T, D)
        """
        super(EndPointSpline, self).__init__()
        B, T, D = xt.shape
        assert t.shape == (T,) and T > 2, "Need at least 3 points"
        assert t.device == xt.device

        t = t.detach().clone()
        xt = xt.permute(1, 0, 2).detach().clone()

        # fix
        self.B = B  # number of (x0,x1) pairs
        self.T = T  # number controlled points / time steps
        self.D = D  # dimension
        self.spline_type = spline_type

        self.register_buffer("t", t)
        self.register_buffer("t_epd", t.reshape(-1, 1).expand(-1, B))
        self.register_buffer("x0", xt[0].reshape(1, B, D))
        self.register_buffer("x1", xt[-1].reshape(1, B, D))
        self.register_parameter("knots", torch.nn.Parameter(xt[1:-1]))

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def xt(self):  # (B, T, D)
        return torch.cat([self.x0, self.knots, self.x1], dim=0).permute(1, 0, 2)

    def interp(self, query_t):
        """
        query_t: (S,) --> yt: (B, S, D)
        """

        (S,) = query_t.shape
        query_t = query_t.reshape(-1, 1).expand(-1, self.B)
        assert query_t.shape == (S, self.B)

        mask = None
        xt = torch.cat([self.x0, self.knots, self.x1], dim=0)  # (T, B, D)
        if self.spline_type == "linear":
            yt = interp1d.linear_interp1d(self.t_epd, xt, mask, query_t)
        elif self.spline_type == "cubic":
            yt = interp1d.cubic_interp1d(self.t_epd, xt, mask, query_t)
        yt = yt.permute(1, 0, 2)
        assert yt.shape == (self.B, S, self.D), yt.shape
        return yt

    def forward(self, t):
        """
        t: (S,) --> yt: (B, S, D)
        """
        return self.interp(t)


class StdSpline(EndPointSpline):
    def __init__(self, t, xt, sigma, spline_type="linear"):
        """
        t: (T,)
        xt: (B, T, 1)
        """
        super(StdSpline, self).__init__(t, xt, spline_type=spline_type)
        assert self.D == 1
        self.sigma = sigma
        self.softplus = torch.nn.Softplus()

    def forward(self, t):
        """
        t: (S,) --> yt: (B, S, 1)
        """
        base = self.sigma * (t * (1 - t)).sqrt()
        xt = self.interp(t)
        return base.reshape(1, -1, 1) * self.softplus(xt)


################################################################################################


class EndPointGaussianPath(torch.nn.Module):
    def __init__(self, t, xt, s, ys, sigma, basedrift):
        super(EndPointGaussianPath, self).__init__()

        (B, T, D), (S,) = xt.shape, s.shape
        assert t.shape == (T,) and ys.shape == (B, S, 1)

        self.B = B  # number of (x0,x1) pairs
        self.T = T  # number controlled points for mean spline
        self.S = S  # number controlled points for std spline
        self.D = D  # dimension

        self.sigma = sigma
        self.mean = EndPointSpline(t, xt)
        self.gamma = StdSpline(s, ys, sigma)
        self.basedrift = basedrift

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def mean_ctl_pts(self):
        return self.mean.xt.detach().cpu()

    @property
    def std_ctl_pts(self):
        return self.gamma(self.gamma.t).detach().cpu()

    def sample_xt(self, t, N):
        """
        N: number of xt for each (x0,x1)
        t: (T,) --> xt: (B, N, T, D)
        """

        mean_t = self.mean(t)  # (B, T, D)
        B, T, D = mean_t.shape

        assert t.shape == (T,)
        std_t = self.gamma(t).reshape(B, 1, T, 1)  # (B, 1, T, 1)

        noise = torch.randn(B, N, T, D, device=t.device)  # (B, N, T, D)

        xt = mean_t.unsqueeze(1) + std_t * noise
        assert xt.shape == noise.shape
        return xt

    def ft(self, t, xt, direction):
        """
        t: (T,)
        xt: (B, N, T, D)
        ===
        ft: (B, N, T, D)
        """
        B, N, T, D = xt.shape
        assert t.shape == (T,)

        sign = 1.0 if direction == "fwd" else -1

        ft = self.basedrift(
            xt.reshape(B * N, T, D),
            t,
        ).reshape(B, N, T, D)
        return sign * ft

    def drift(self, t, xt, direction):
        """Implementation of the drift of Gaussian path in Eq 8
        t: (T,)
        xt: (B, N, T, D)
        ===
        drift: (B, N, T, D)
        """
        assert (t > 0).all() and (t < 1).all()

        B, N, T, D = xt.shape
        assert t.shape == (T,)

        mean, dmean = torch.autograd.functional.jvp(
            self.mean, t, torch.ones_like(t), create_graph=self.training
        )
        assert mean.shape == dmean.shape == (B, T, D)

        dmean = dmean.reshape(B, 1, T, D)
        mean = mean.reshape(B, 1, T, D)

        std, dstd = torch.autograd.functional.jvp(
            self.gamma, t, torch.ones_like(t), create_graph=self.training
        )
        assert std.shape == dstd.shape == (B, T, 1)

        if direction == "fwd":
            # u = ∂m + a (x - m),
            # a = (\dot γ - σ^2 / 2γ) / γ
            #   = -1 / (1-t), if γ is the std of brownian bridge
            a = (dstd - self.sigma**2 / (2 * std)) / std
            if self.sigma == 0:
                a = torch.zeros_like(a)  # handle deterministic cases
            drift = dmean + a.reshape(B, 1, T, 1) * (xt - mean)
        else:
            # u = -∂m + a (x - m),
            # a = (-\dot γ - σ^2 / 2γ) / γ
            #   = -1 / t, if γ is the std of brownian bridge
            a = (-dstd - self.sigma**2 / (2 * std)) / std
            if self.sigma == 0:
                a = torch.zeros_like(a)  # handle deterministic cases
            drift = -dmean + a.reshape(B, 1, T, 1) * (xt - mean)

        assert drift.shape == xt.shape
        return drift

    def ut(self, t, xt, direction):
        """
        t: (T,)
        xt: (B, N, T, D)
        ===
        ut: (B, N, T, D)
        """
        ft = self.ft(t, xt, direction)
        drift = self.drift(t, xt, direction)
        assert drift.shape == ft.shape == xt.shape
        return drift - ft

    def forward(self, t, N, direction):
        """
        t: (T,)
        ===
        xt: (B, N, T, D)
        ut: (B, N, T, D)
        """
        xt = self.sample_xt(t, N)

        B, N, T, D = xt.shape
        assert t.shape == (T,)

        ut = self.ut(t, xt, direction)
        assert ut.shape == xt.shape

        return xt, ut


################################################################################################


def build_loss_fn(gpath, sigma, V, ccfg):
    def loss_fn(t, xt, ut):
        B, N, T, D = xt.shape
        assert t.shape == (T,) and ut.shape == (B, N, T, D)

        cost_s = V(xt, t, gpath).reshape(B, N, T)
        scale = (0.5 / (sigma**2)) if ccfg.scale_by_sigma else 0.5
        cost_c = scale * (ut**2).sum(dim=-1)
        assert cost_s.shape == cost_c.shape == (B, N, T)
        return (cost_s + cost_c).mean()

    return loss_fn


def build_img_loss_fn(gpath, sigma, V, ccfg):

    ### define inputs for VAE
    x0 = gpath.mean.xt[:, 0].detach()
    x1 = gpath.mean.xt[:, -1].detach()
    recon_xt, _ = V.latent_interp(x0, x1, ccfg.S)

    def loss_fn(t, xt, ut):
        B, N, T, D = xt.shape
        assert T == ccfg.S
        assert t.shape == (T,) and ut.shape == (B, N, T, D)

        cost_s = V(xt, t, recon_xt.detach()).reshape(B, N, T)
        scale = (0.5 / (sigma**2)) if ccfg.scale_by_sigma else 0.5
        cost_c = scale * (ut**2).mean(dim=-1)
        assert cost_s.shape == cost_c.shape == (B, N, T)
        return (ccfg.weight_s * cost_s + ccfg.weight_c * cost_c).mean()

    return loss_fn


def build_optim(gpath, ccfg):
    if ccfg.optim == "sgd":
        return torch.optim.SGD(
            [
                {"params": gpath.mean.parameters(), "lr": ccfg.lr_mean},
                {"params": gpath.gamma.parameters(), "lr": ccfg.lr_gamma},
            ],
            momentum=ccfg.momentum,
        )
    elif ccfg.optim == "adam":
        return torch.optim.Adam(
            [
                {"params": gpath.mean.parameters(), "lr": ccfg.lr_mean},
                {"params": gpath.gamma.parameters(), "lr": ccfg.lr_gamma},
            ],
        )
    else:
        raise ValueError(f"Unsupported Spline optimizer {ccfg.optim}!")


def fit(ccfg, gpath, direction, loss_fn, eps=0.001, verbose=False):
    """
    V: xt: (*, T, D), t: (T,), gpath --> (*, T)
    """
    assert direction in DIRECTIONS

    results = {"name": ccfg.name}
    results["init_mean"] = gpath.mean_ctl_pts
    results["init_gamma"] = gpath.std_ctl_pts

    ### setup
    B, D, N, T, device = gpath.B, gpath.D, ccfg.N, ccfg.S, gpath.device
    optim = build_optim(gpath, ccfg)

    ### optimize spline
    gpath.train()
    losses = np.zeros(ccfg.nitr)
    bar = trange(ccfg.nitr) if verbose else range(ccfg.nitr)
    for itr in bar:
        optim.zero_grad()

        t = torch.linspace(eps, 1 - eps, T, device=device)
        xt, ut = gpath(t, N, direction)
        assert xt.shape == ut.shape == (B, N, T, D)

        loss = loss_fn(t, xt, ut)

        loss.backward()
        optim.step()
        losses[itr] = loss.cpu().item()
        if verbose:
            bar.set_description(f"loss={losses[itr]}")

    gpath.eval()

    results["final_mean"] = gpath.mean_ctl_pts
    results["final_gamma"] = gpath.std_ctl_pts
    results["gpath"] = copy.deepcopy(gpath).cpu()
    results["losses"] = losses

    return results
