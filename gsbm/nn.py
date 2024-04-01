"""
Various utilities for neural networks.
Taken from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
"""

import math

import torch as th
import torch.nn as nn


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class MaskMixin:
    pass


class MaskedConv1d(nn.Conv1d, MaskMixin):
    def forward(self, x, mask=None, **kwargs):
        if mask is not None:
            x = x * mask
        return super().forward(x)


class MaskedConv2d(nn.Conv2d, MaskMixin):
    def forward(self, x, mask=None, **kwargs):
        if mask is not None:
            x = x * mask
        return super().forward(x)


class MaskedConv3d(nn.Conv3d, MaskMixin):
    def forward(self, x, mask=None, **kwargs):
        if mask is not None:
            x = x * mask
        return super().forward(x)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return MaskedConv1d(*args, **kwargs)
    elif dims == 2:
        return MaskedConv2d(*args, **kwargs)
    elif dims == 3:
        return MaskedConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class LayerNorm(th.nn.LayerNorm):
    def forward(self, x):
        d = x.ndim - 1
        return super().forward(x.transpose(1, d)).transpose(1, d)


def normalization(channels, normalization_type: str = "group_norm"):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if normalization_type == "group_norm":
        return GroupNorm32(32, channels)
    elif normalization_type == "layer_norm":
        return LayerNorm(channels)
    else:
        raise ValueError("Unknown normalization type!")


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, kwargs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        # Use pytorch's activation checkpointing.  This has support for fp16 autocast
        return th.utils.checkpoint.checkpoint(func, *inputs, **kwargs)
        # args = tuple(inputs) + tuple(params)
        # return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs, **kwargs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class Unbatch(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, *args, **kwargs):
        has_batch = x.ndim > 1
        if not has_batch:
            x = x.reshape(1, -1)  # (1,nx)
            t = t.reshape(-1)  # (1,)
        v = self.net(x, t, *args, **kwargs)
        if not has_batch:
            v = v[0]
        return v


class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map = nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)]
        )

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths = [hid] * 4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / 2
        return h
