"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def load_model(checkpoint, device=None):
    chkpnt = torch.load(checkpoint, map_location="cpu")

    # Get just the model state dict.
    sd = chkpnt["state_dict"]
    new_sd = {}
    for key in sd.keys():
        if key.startswith("model."):
            new_sd[key[6:]] = sd[key]

    try:
        cfg = chkpnt["cfg"]
        model = VAE(z_dim=cfg.z_dim, beta=cfg.beta, x_std=cfg.x_std)
    except:
        model = VAE(z_dim=256)
    model.load_state_dict(new_sd)
    if device is not None:
        model.to(device)

    # Important since there is batch norm.
    model.eval()

    return model


def lerp(x0, x1, t):
    """Assumes all inputs can be broadcasted to the same shape."""
    return x0 + t * (x1 - x0)


def slerp(x0, x1, t):
    """
    Assumes all inputs can be broadcasted to the same shape (..., D).
    Performs the slerp on the last dimension.
    """
    low_norm = x0 / torch.norm(x0, dim=-1, keepdim=True)
    high_norm = x1 / torch.norm(x1, dim=-1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - t) * omega) / so).unsqueeze(-1) * x0 + (
        torch.sin(t * omega) / so
    ).unsqueeze(-1) * x1
    return res


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return x * torch.sigmoid_(x * F.softplus(self.beta))


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1, same_width=False):
        super().__init__()

        if same_width:
            planes = in_planes
        else:
            planes = in_planes * stride

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.actfn1 = Swish()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.actfn2 = Swish()

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.actfn1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.actfn2(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1, same_width=False):
        super().__init__()

        if same_width:
            planes = in_planes
        else:
            planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.actfn2 = Swish()

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )
        self.actfn1 = Swish()

    def forward(self, x):
        out = self.actfn2(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = self.actfn1(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3, additional_layers=0):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.actfn1 = Swish()
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

        if additional_layers > 0:
            layers = [
                self._make_layer(
                    BasicBlockEnc, 512, num_Blocks[3], stride=2, same_width=True
                )
                for _ in range(additional_layers)
            ]
            self.additional_layers = nn.Sequential(*layers)
        else:
            self.additional_layers = None

        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride, same_width=False):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride, same_width=same_width)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.actfn1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.additional_layers is not None:
            x = self.additional_layers(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, : self.z_dim]
        logstd = x[:, self.z_dim :]
        return mu, logstd


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3, additional_layers=0):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        if additional_layers > 0:
            layers = [
                self._make_layer(
                    BasicBlockDec, 512, num_Blocks[3], stride=2, same_width=True
                )
                for _ in range(additional_layers)
            ]
            self.additional_layers = nn.Sequential(*layers)
        else:
            self.additional_layers = None

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride, same_width=False):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride, same_width=same_width)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        if self.additional_layers is not None:
            x = self.additional_layers(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.tanh(self.conv1(x))  # squash output to [-1, 1]
        return x


class VAE(nn.Module):
    def __init__(self, z_dim, beta=1.0, x_std=0.1, additional_layers=0):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim, additional_layers=additional_layers)
        self.decoder = ResNet18Dec(z_dim=z_dim, additional_layers=additional_layers)
        self.z_dim = z_dim
        self.beta = beta
        self.logsigma = math.log(x_std)  # logstd of p(x | z).

    def sample_latent(self, x):
        mean, logstd = self.encoder(x)
        z = torch.randn_like(mean) * torch.exp(logstd) + mean
        return z, mean, logstd

    def forward(self, x):
        z, mean, logstd = self.sample_latent(x)
        x = self.decoder(z)
        return x, z, mean, logstd

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decoder(z)

    def reconstruct(self, x):
        return self(x)[0]

    def compute_elbo(self, x):
        """Computes the ELBO."""
        bsz = x.shape[0]
        x_recon, z, mean, logstd = self(x)

        logqz = normal_logprob(z, mean, logstd).reshape(bsz, -1).sum(1)
        logpz = normal_logprob(z, 0.0, 0.0).reshape(bsz, -1).sum(1)
        logpx = normal_logprob(x, x_recon, self.logsigma).reshape(bsz, -1).sum(1)

        return logpx + self.beta * (logpz - logqz)


def normal_logprob(z, mean, log_std):
    mean = (mean + torch.tensor(0.0)).to(z)
    log_std = (log_std + torch.tensor(0.0)).to(z)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)
