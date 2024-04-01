"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import math
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import get_repo_path
from .state_cost import LIDARStateCost
from ipdb import set_trace as debug


# make transforms.Lambda(lambda x: x * 2 - 1) picklable
class Normalize(object):
    def __call__(self, img):
        return img * 2 - 1

    def __repr__(self):
        return self.__class__.__name__


class AFHQ(Dataset):
    animals = ["cat", "dog", "wild"]

    def __init__(self, resize, animals, split="train"):
        assert split in ("train", "val")

        np_imgs = []
        for ani in animals:
            assert ani in self.animals
            out = np.load(get_repo_path() / "data" / f"afhq{resize}-{split}-{ani}.npz")
            np_imgs.append(out["data"])

        np_imgs = np.concatenate(np_imgs, axis=0)
        th_imgs = torch.from_numpy(np_imgs) / 255.0  # [0, 1]

        self.th_imgs = th_imgs.permute(0, 3, 1, 2)
        transform_list = (
            [
                transforms.RandomHorizontalFlip(),
            ]
            if split == "train"
            else []
        )
        transform_list.append(Normalize())
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.th_imgs.shape[0]

    def __getitem__(self, idx):
        image_tensor = self.transform(self.th_imgs[idx])
        return image_tensor


class ImageSampler:
    def __init__(self, dataset, generator=None):
        self.dataset = dataset
        self.generator = generator

    def set_generator(self, generator):
        self.generator = generator

    def __call__(self, n):
        ii = torch.randint(0, len(self.dataset), (n,), generator=self.generator)
        out = torch.stack([self.dataset[i] for i in ii], dim=0)
        return out.reshape(n, -1)


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


class Gaussian:
    def __init__(self, mu, var, generator):
        self.mean = torch.tensor(mu).float()
        self.std = torch.tensor(var).sqrt()
        self.generator = generator

    def set_generator(self, generator):
        self.generator = generator

    def __call__(self, n):
        noise_shape = (n,) + self.mean.shape
        return (
            torch.randn(*noise_shape, generator=self.generator).to(self.mean) * self.std
            + self.mean
        )


class Opinion(Gaussian):
    def __init__(self, dim, mu, var, var_1st_dim, generator=None):
        mu = mu * torch.ones(dim)
        var = var * torch.ones(dim)
        if var_1st_dim is not None:
            var[0] = var_1st_dim
        super(Opinion, self).__init__(mu, var, generator)


class GaussianMM:
    def __init__(self, mu, var, generator=None):
        super().__init__()
        self.centers = torch.tensor(mu)
        self.logstd = torch.tensor(var).log() / 2.0
        self.K = self.centers.shape[0]
        self.generator = generator

    def set_generator(self, generator):
        self.generator = generator

    def logprob(self, x):
        """Computes the log probability."""
        logprobs = normal_logprob(
            x.unsqueeze(1), self.centers.unsqueeze(0), self.logstd
        )
        logprobs = torch.sum(logprobs, dim=2)
        return torch.logsumexp(logprobs, dim=1) - math.log(self.K)

    def __call__(self, n_samples):
        idx = torch.randint(self.K, (n_samples,)).to(self.centers.device)
        mean = self.centers[idx]
        return (
            torch.randn(*mean.shape, generator=self.generator).to(mean)
            * torch.exp(self.logstd)
            + mean
        )


class LiDARProjector:
    """Takes an existing dataset and projects all points onto the manifold."""

    def __init__(self, dataset, lcfg):
        self.manifold = LIDARStateCost(lcfg)
        self.dataset = dataset

    def set_generator(self, generator):
        self.dataset.set_generator(generator)

    def __call__(self, n_samples):
        samples = self.dataset(n_samples)
        projx = self.manifold.get_tangent_proj(samples)
        samples = projx(samples)
        return samples


class PairDataset(Dataset):
    def __init__(self, x0, x1, expand_factor=1):
        assert len(x0) == len(x1)
        self.x0 = x0
        self.x1 = x1
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.x0) * self.expand_factor

    def __getitem__(self, idx):
        return {"x0": self.x0[idx % len(self.x0)], "x1": self.x1[idx % len(self.x0)]}


class SplineDataset(Dataset):
    def __init__(self, mean_t, mean_xt, gamma_s, gamma_xs, expand_factor=1):
        """
        mean_t: (T,)
        mean_xt: (B, T, D)
        gamma_t: (S,)
        gamma_xt: (B, S, 1)
        """
        (B, T, D), (S,) = mean_xt.shape, gamma_s.shape
        assert T > 3 and S > 3
        assert mean_t.shape == (T,)
        assert gamma_xs.shape == (B, S, 1)

        self.mean_t = mean_t.detach().cpu().clone()
        self.mean_xt = mean_xt.detach().cpu().clone()
        self.gamma_s = gamma_s.detach().cpu().clone()
        self.gamma_xs = gamma_xs.detach().cpu().clone()

        self.expand_factor = expand_factor

    def __len__(self):
        return self.mean_xt.shape[0] * self.expand_factor

    def __getitem__(self, idx):
        _idx = idx % self.mean_xt.shape[0]

        x0 = self.mean_xt[_idx, 0]
        x1 = self.mean_xt[_idx, -1]
        mean_xt = self.mean_xt[_idx]
        gamma_xs = self.gamma_xs[_idx]

        return {
            "x0": x0,
            "x1": x1,
            "mean_t": self.mean_t,
            "mean_xt": mean_xt,
            "gamma_s": self.gamma_s,
            "gamma_xs": gamma_xs,
        }


class SplineIWDataset(Dataset):
    def __init__(self, spline_ds, IW_t, IW_xs, weights):
        """
        IW_t: (TT,)
        IW_xs: (B, N, TT, D)
        weights: (B, N)
        """
        B, N, TT, D = IW_xs.shape
        assert weights.shape == (
            B,
            N,
        )
        assert IW_t.shape == (TT,)

        self.IW_t = IW_t.detach().cpu().clone()
        self.IW_xs = IW_xs.detach().cpu().clone()
        self.weights = weights.detach().cpu().clone()

        self.expand_factor = spline_ds.expand_factor
        spline_ds.expand_factor = 1
        self.spline_ds = spline_ds
        assert len(self.spline_ds) == B
        assert spline_ds.mean_xt.shape[2] == D

    def __len__(self):
        return self.IW_xs.shape[0] * self.expand_factor

    def __getitem__(self, idx):
        _idx = idx % self.IW_xs.shape[0]

        out = self.spline_ds.__getitem__(_idx)
        out["IW_t"] = self.IW_t
        out["IW_xs"] = self.IW_xs[_idx]
        out["weights"] = self.weights[_idx]
        return out


def get_sampler(p, gen=None, **kwargs):
    name = p.name

    if name == "gaussian":
        return Gaussian(p.mu, p.var, generator=gen)
    elif name == "gmm":
        return GaussianMM(p.mu, p.var, generator=gen)
    elif name == "opinion":
        return Opinion(p.dim, p.mu, p.var, p.get("var_1st_dim", None), generator=gen)
    elif name == "lidarproj":
        dataset = GaussianMM(p.mu, p.var, generator=gen)
        return LiDARProjector(dataset, p.lcfg)
    elif name == "afhq":
        dataset = AFHQ(resize=p.resize, animals=p.animals, **kwargs)
        return ImageSampler(dataset)
    else:
        raise ValueError(f"Unknown distribution option: {name}")


def get_dist_boundary(cfg):
    p0 = get_sampler(cfg.prob.p0)
    p1 = get_sampler(cfg.prob.p1)
    p0_val = get_sampler(cfg.prob.p0, split="val")
    p1_val = get_sampler(cfg.prob.p1, split="val")
    return p0, p1, p0_val, p1_val
