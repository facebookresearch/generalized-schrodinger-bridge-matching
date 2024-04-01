"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributions as td

from geomloss import SamplesLoss
from ot.sliced import sliced_wasserstein_distance
from .state_cost import build_obstacle_cost, congestion_cost, zero_cost_fn
from .utils import get_repo_path

from ipdb import set_trace as debug


def build_evaluator(cfg):
    if cfg.prob.name in ["opinion", "afhq", "lidar"]:
        return DumpEvaluator()
    else:
        return CrowdNavEvaluator(cfg)


def cpu_everything(*args):
    return [a.cpu() for a in args] if len(args) > 1 else args.cpu()


def shuffle(t):
    """
    t: (B, *) --> (B, *)
    """
    return t[torch.randperm(t.shape[0])]


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


@torch.no_grad()
def est_entropy_cost(xt, std=0.2):
    """
    xt: (B, T, D) --> (B, T)
    """
    B, T, D = xt.shape

    ### build B*T indep Gaussians with given std / bandwidth
    normals = td.Normal(
        xt.reshape(B * T, D),
        std * torch.ones(B * T, D).to(xt),
    )
    indep_normals = td.Independent(normals, 1)

    ### evaluate log-prob of all `B` samples at each timestamp
    ### w.r.t. `B` Gaussians
    xxt = xt.unsqueeze(1).expand(-1, B, -1, -1)
    assert xxt.shape == (B, B, T, D)

    log_pt_01 = indep_normals.log_prob(xxt.reshape(B, B * T, D)).reshape(B, B, T)
    pt = log_pt_01.exp().mean(dim=1)  # (B, T)

    log_pt = pt.log()
    assert not torch.isnan(log_pt).any()
    assert log_pt.shape == (B, T)
    return log_pt


##########################################################


class DumpEvaluator:
    def __call__(self, samples):
        return {}


class CrowdNavEvaluator:
    B = 1000
    D = 2

    def __init__(self, cfg) -> None:
        self.ccfg = cfg.csoc
        self.scfg = cfg.state_cost
        self.sigma = cfg.prob.sigma

        self.obstacle_cost = build_obstacle_cost(cfg.prob.name)
        self.sinkhorn_cfg = {"p": 2, "blur": 0.05, "scaling": 0.95}
        self.ref_x0, self.ref_x1 = self.build_ref_x(cfg)

    def build_ref_x(self, cfg):
        ref_fn = get_repo_path() / "data" / f"{cfg.prob.name}.pt"

        if not ref_fn.exists():
            from .dataset import get_sampler

            ref_x0 = get_sampler(cfg.prob.p0)(self.B)
            ref_x1 = get_sampler(cfg.prob.p1)(self.B)
            torch.save({"ref_x0": ref_x0, "ref_x1": ref_x1}, ref_fn)
            print(f"Saved new reference file to {ref_fn}!")
            return ref_x0, ref_x1
        else:
            ref_pt = torch.load(ref_fn, map_location="cpu")
            return ref_pt["ref_x0"], ref_pt["ref_x1"]

    def boundary_metrics(self, xs):
        ## Resmple batch dimension if needed
        B, T, D = xs.shape
        if B < self.B:
            rand_idx = torch.randint(0, B, (self.B,))
            xs = xs[rand_idx]
        elif B > self.B:
            rand_idx = torch.randperm(B)[: self.B]
            xs = xs[rand_idx]
        assert xs.shape == (self.B, T, D)

        ## Build x0, x1
        x0, x1 = shuffle(xs[:, 0]), shuffle(xs[:, -1])
        assert x0.shape == self.ref_x0.shape == x1.shape == self.ref_x1.shape

        ## Compute metrics
        metrics = dict()
        metrics["SWD_0"] = sliced_wasserstein_distance(x0, self.ref_x0)
        metrics["SWD_1"] = sliced_wasserstein_distance(x1, self.ref_x1)

        sinkhorn = SamplesLoss("sinkhorn", **self.sinkhorn_cfg)
        metrics["Sinkhorn_0"] = sinkhorn(x0, self.ref_x0)
        metrics["Sinkhorn_1"] = sinkhorn(x1, self.ref_x1)

        mmd = MMD_loss()
        metrics["MMD_0"] = mmd(x0, self.ref_x0)
        metrics["MMD_1"] = mmd(x1, self.ref_x1)
        return metrics

    def state_costs(self, xs):
        (B, T, D), scfg = xs.shape, self.scfg
        assert "obs" in scfg.type and scfg.obs > 0

        cost_s = scfg.obs * self.obstacle_cost(xs)
        if "ent" in scfg.type and scfg.ent > 0:
            cost_s = cost_s + scfg.ent * est_entropy_cost(xs)
        elif "cgst" in scfg.type and scfg.cgst > 0:
            cost_s = cost_s + scfg.cgst * congestion_cost(xs)

        assert cost_s.shape == (B, T)
        return cost_s

    def cost_metrics(self, xs, us):
        B, T, D = xs.shape
        assert us.shape == (B, T, D)

        scale = (0.5 / (self.sigma**2)) if self.ccfg.scale_by_sigma else 0.5
        cost_c = scale * (us**2).sum(dim=-1)
        cost_s = self.state_costs(xs)
        assert cost_c.shape == cost_s.shape == (B, T)

        metrics = dict()
        metrics["control_cost"] = cost_c.mean()
        metrics["state_cost"] = cost_s.mean()
        metrics["total_cost"] = metrics["control_cost"] + metrics["state_cost"]
        return metrics

    def __call__(self, samples):
        """
        xs: (B, T, D)
        us: (B, T, D)
        """
        xs, us = cpu_everything(samples["xs"], samples["us"])
        B, T, D = xs.shape
        assert us.shape == (B, T, D)

        metrics = {}
        metrics.update(self.boundary_metrics(xs))
        metrics.update(self.cost_metrics(xs, us))
        for k, v in metrics.items():
            metrics[k] = v.item()
        return metrics
