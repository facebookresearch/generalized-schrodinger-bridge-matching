"""
Modified from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
    MaskMixin,
)
from ipdb import set_trace as debug

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb=None, mask=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, **kwargs):
        for layer in self:
            if isinstance(layer, (TimestepBlock, MaskMixin)):
                x = layer(x, **kwargs)
            else:
                x = layer(x)

            if isinstance(x, tuple):
                x, mask = x
                kwargs["mask"] = mask
        return x, kwargs.get("mask")


class Upsample(nn.Module, MaskMixin):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x, size=None, mask=None, new_mask=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            assert mask is None, "Mask not implemented yet!"
            upsampled = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            kwargs = {"scale_factor": 2}
            if size:
                kwargs = {"size": size}
            if new_mask is not None:
                raise NotImplementedError()
                from extensions.interpolate import interpolate

                source_sizes = mask.squeeze(1).sum(-1)
                tgt_sizes = new_mask.squeeze(1).sum(-1)
                upsampled = interpolate(x, source_sizes, tgt_sizes)
            else:
                upsampled = F.interpolate(x, mode="nearest", **kwargs)
        if self.use_conv:
            assert mask is None, "Mask not implemented yet!"
            upsampled = self.conv(x)
        return upsampled, (mask if new_mask is None else new_mask)


class Downsample(nn.Module, MaskMixin):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.use_conv = use_conv
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, size=None, mask=None, **kwargs):
        assert x.shape[1] == self.channels
        downsampled = self.op(x)
        downsampled_mask = mask
        if mask is not None:
            assert not self.use_conv, "Masking not implemented for use_conv!"
            downsampled_mask = self.op(mask.float()) == 1
        return downsampled, downsampled_mask


def identity_map(x, **kwargs):
    return x


class ResBlock(TimestepBlock, MaskMixin):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        normalization_type="group_norm",
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = TimestepEmbedSequential(
            normalization(channels, normalization_type),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = TimestepEmbedSequential(
            normalization(self.out_channels, normalization_type),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = identity_map  # lambda x, **kwargs: x
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, **kwargs):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward,
            (x, emb, kwargs.get("mask"), kwargs.get("size"), kwargs.get("new_mask")),
            {},
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x, emb, mask=None, size=None, new_mask=None, **kwargs):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h, mask = in_rest(x, mask=mask)
            h, _ = self.h_upd(h, size=size, mask=mask, new_mask=new_mask)
            x, mask = self.x_upd(x, size=size, mask=mask, new_mask=new_mask)
            h = in_conv(h, mask=mask)
        else:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h, mask = in_rest(x, mask=mask)
            h = in_conv(h, mask=mask)
            h, mask = self.in_layers(x, mask=mask)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h, mask = out_rest(h, mask=mask)
        else:
            h = h + emb_out
            h, mask = self.out_layers(h, mask=mask)
        return self.skip_connection(x, mask=mask) + h, mask


class AttentionBlock(nn.Module, MaskMixin):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        memory_efficient_attention=False,
        normalization_type="group_norm",
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, normalization_type)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        elif memory_efficient_attention:
            raise NotImplementedError()
            # self.attention = MemoryEfficientAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, mask=None, **kwargs):
        return checkpoint(
            self._forward, (x, mask), {}, self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, mask=None, **kwargs):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x), mask=mask)
        h = self.attention(qkv, mask=mask)
        h = self.proj_out(h, mask=mask)
        return (x + h).reshape(b, c, *spatial), mask


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])



class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards

        weight = weight.float()
        if mask is not None:
            # Mask is true for non-pad, and false for pad values.  Invert so that we can fill in
            # The padded entries with -inf
            # https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/utils.py#L22
            seq_len = mask.size(-1)
            mask = ~mask.squeeze(1)  # B x S
            mask = mask.unsqueeze(1) | mask.unsqueeze(-1)  # B x S x S
            mask = (
                mask.unsqueeze(1)
                .expand(-1, self.n_heads, -1, -1)
                .reshape(-1, seq_len, seq_len)
            )
            weight.masked_fill_(mask, th.finfo(weight.dtype).min)

        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards

        if mask is not None:
            # Mask is true for non-pad, and false for pad values.  Invert so that we can fill in
            # The padded entries with -inf
            # https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/utils.py#L22
            seq_len = mask.size(-1)
            mask = ~mask.squeeze(1)  # B x S
            mask = mask.unsqueeze(1) | mask.unsqueeze(-1)  # B x S x S
            mask = (
                mask.unsqueeze(1)
                .expand(-1, self.n_heads, -1, -1)
                .reshape(-1, seq_len, seq_len)
            )
            weight.masked_fill_(mask, th.finfo(weight.dtype).min)

        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


@dataclass(eq=False)
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param embedding_dim: conditioning stream embedding dimension
    :param mask_embedding_dim: mask indicator stream embedding dimension
    """

    in_channels: int
    model_channels: int = 128
    out_channels: int = 3
    embedding_dim: Optional[int] = None
    mask_embedding_dim: int = 0
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1, 2, 2, 2)
    dropout: float = 0.0
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    with_fourier_features: bool = False
    use_new_attention_order: bool = True
    use_memory_efficient_attention: bool = False
    normalization_type: str = "group_norm"

    n_embeddings: Optional[int] = None

    image_size: int = -1  # not used...
    _target_: str = "fair_diffusion.models.gd_unet.UNetModel"

    def __post_init__(self):
        super().__init__()

        if self.n_embeddings is not None:
            self.embeddings = th.nn.Embedding(
                num_embeddings=self.n_embeddings + 1,
                embedding_dim=(
                    self.embedding_dim
                    if self.embedding_dim is not None
                    else self.out_channels
                ),
                padding_idx=self.n_embeddings,
            )

        if self.mask_embedding_dim > 0:
            self.mask_embeddings = th.nn.Embedding(
                num_embeddings=2,
                embedding_dim=self.mask_embedding_dim,
            )

        if self.with_fourier_features:
            self.in_channels += 12

        if self.num_heads_upsample == -1:
            self.num_heads_upsample = self.num_heads

        self.time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, self.time_embed_dim)

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(self.dims, self.in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        normalization_type=self.normalization_type,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            memory_efficient_attention=self.use_memory_efficient_attention,
                            normalization_type=self.normalization_type,
                        )  # type: ignore
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            normalization_type=self.normalization_type,
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                normalization_type=self.normalization_type,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
                memory_efficient_attention=self.use_memory_efficient_attention,
                normalization_type=self.normalization_type,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                self.dropout,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                normalization_type=self.normalization_type,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        normalization_type=self.normalization_type,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            memory_efficient_attention=self.use_memory_efficient_attention,
                            normalization_type=self.normalization_type,
                        )  # type: ignore
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            normalization_type=self.normalization_type,
                        )
                        if self.resblock_updown
                        else Upsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                        )  # type: ignore
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = TimestepEmbedSequential(
            normalization(ch, self.normalization_type),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def prepare_input(self, batch):
        x = []
        if "noisy_x" in batch:
            x.append(batch["noisy_x"])
        if self.with_fourier_features:
            z_f = base2_fourier_features(x, start=6, stop=8, step=1)
            x.append(z_f)

        if "phonemes" in batch and "alignment" not in batch:
            assert "masked_duration" in batch
            phone_emb = self.embeddings(batch["phonemes"].to(th.int)).transpose(1, 2)
            x.append(phone_emb)

        if "alignment" in batch and "phonemes" in batch:
            phonemes = batch["phonemes"].gather(1, batch["alignment"])
            phone_emb = self.embeddings(phonemes).transpose(1, 2)
            x.append(phone_emb)

        if "masked_spectrogram" in batch:
            batch["concat_conditioning"] = batch["masked_spectrogram"]

        if "masked_duration" in batch:
            batch["concat_conditioning"] = batch["masked_duration"]

        if "concat_conditioning" in batch:
            x.append(batch["concat_conditioning"])

        if self.mask_embedding_dim > 0:
            mask_inp = (
                (
                    batch["spectrogram_mask"]
                    if "alignment" in batch
                    else batch["duration_mask"]
                )
                .squeeze(1)
                .to(th.int)
            )
            mask_emb = self.mask_embeddings(mask_inp.to(th.int))
            mask_emb = mask_emb.transpose(1, 2)
            x.append(mask_emb)

        return th.cat(x, dim=1)

    def forward(self, batch, timesteps=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        x = self.prepare_input(batch)

        if timesteps is None:
            timesteps = th.zeros((x.size(0),), device=x.device)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        scaler = 1.0
        if self.num_classes and "y" not in batch:
            # Hack to deal with ddp find_unused_parameters not working with activation checkpointing...
            # self.num_classes corresponds to the pad index of the embedding table
            batch["y"] = th.full((x.size(0),), 0, dtype=th.long, device=x.device)
            scaler = 0.0

        if self.num_classes is not None and "y" in batch:
            assert batch["y"].shape == (x.shape[0],)
            emb = emb + self.label_emb(batch["y"]) * scaler
        h = x

        mask = batch.get("mask")
        if (
            mask is not None
            and self.normalization_type != "layer_norm"
            and not mask.all()
        ):
            raise ValueError(
                f"Variable length sequences are only supported with layer_norm, not: {self.normalization_type}"
            )

        for module in self.input_blocks:
            h = module(h, emb=emb, mask=mask)
            if isinstance(h, tuple):
                h, mask = h
            hs.append((h, mask))

        h, mask = self.middle_block(h, emb=emb, mask=mask)

        for i, module in enumerate(self.output_blocks):
            skip_h, _ = hs.pop()
            h = th.cat([h, skip_h], dim=1)
            spatial_shape = hs[-1][0].shape[2:] if len(hs) > 0 else None
            kwargs = {}
            if spatial_shape and spatial_shape[-1] != h.size(-1):
                # We are about to upsample, so pass in the proper resolution mask from the downsampling pass
                kwargs["new_mask"] = hs[-1][1]
            h, mask = module(h, emb=emb, size=spatial_shape, mask=mask, **kwargs)

        h = h.type(x.dtype)

        res, _ = self.out(h, mask=mask)
        return res


# Based on https://github.com/google-research/vdm/blob/main/model_vdm.py
def base2_fourier_features(
    inputs: th.Tensor, start: int = 0, stop: int = 8, step: int = 1
) -> th.Tensor:
    freqs = th.arange(start, stop, step, device=inputs.device, dtype=inputs.dtype)

    # Create Base 2 Fourier features
    w = 2.0**freqs * 2 * np.pi
    w = th.tile(w[None, :], (1, inputs.size(1)))

    # Compute features
    h = th.repeat_interleave(inputs, len(freqs), dim=1)
    h = w[:, :, None, None] * h
    h = th.cat([th.sin(h), th.cos(h)], dim=1)
    return h
