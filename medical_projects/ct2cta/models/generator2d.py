# !/usr/bin/env python3
# coding=utf-8
#
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.cuda.amp import autocast, GradScaler

from .utils import get_norm_layer, weights_init, count_parameters


class Generator(nn.Module):
    """
    generator
    """
    def __init__(self, in_size=512, model_type="stylegan2", use_dropout=False, use_amp=False,
                 generator_scale=1.0, use_checkpoint=False):
        """
        init
        Args:
            in_size:
            model_type:
            use_dropout:
            use_amp:
            generator_scale:
            use_checkpoint:
        """
        super(Generator, self).__init__()

        self.model_type = model_type
        self.in_size = in_size
        self.use_dropout = use_dropout
        self.use_amp = use_amp

        assert self.model_type in ["unet", "stylegan2"]
        assert self.in_size in [512]

        self.netG = Generator2D(size=self.in_size, channel_multiplier=2, generator_scale=generator_scale)

    def forward(self, x):
        """
        forward
        Args:
            x:
        Returns:
        """
        if self.use_amp:
            with autocast():
                return self.netG(x)
        else:
            return self.netG(x)


class Identity(nn.Module):
    """
    identity
    """
    def __init__(self):
        """
        init
        """
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Args:
            x:
        Returns:
        """
        return x


class Blur(nn.Module):
    """
    blur ops follow official tensorflow code, but use depthwise conv
    """

    def __init__(self, pad, normalize=True, stride=1):
        """
        init
        :param normalize: whether normalize
        :param stride: blur stride
        :param pad: (left_w, right_w, top_h, bottom_h)
                    stride 1 pad (1, 1, 1, 1)
                    stride 2 pad (1, 2, 1, 2)
        """
        super(Blur, self).__init__()
        kernel = [1, 3, 3, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]

        if normalize:
            kernel = kernel / kernel.sum()

        self.register_buffer('kernel', kernel)
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.pad(x, [self.pad[0], self.pad[1], self.pad[2], self.pad[3]])
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            groups=x.size(1)
        )
        return x


class Upscale2d(nn.Module):
    """
    upscale 2d follow official tensorflow code, like a strict nearest upsample
    """

    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        """
        core func
        :param factor: upscale factor
        :param gain: gain for input
        """
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        """
        init
        :param factor: upscale factor
        :param gain: gain for input
        """
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class EqualConv2d(nn.Module):
    """
    equal conv2d follow paper
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        """
        init
        :param in_channel: input channel
        :param out_channel: output channel
        :param kernel_size: conv kernel size
        :param stride: conv stride
        :param padding: conv padding
        :param bias: conv bias
        """
        super(EqualConv2d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
            # torch.from_numpy(
            #     np.random.randn(out_channel, in_channel, kernel_size, kernel_size)
            # ).type(torch.float32)  # 对齐paddle时测试权重
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        """
        forward func
        :param input: input
        """
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            )

        return out


class EqualLinear(nn.Module):
    """
    equal linear follow paper
    """

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        """
        init
        :param in_dim: input channel
        :param out_dim: output channel
        :param bias: linear bias
        :param bias_init: bias init val
        :param lr_mul: linear param mul val
        """
        super(EqualLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        # self.weight = nn.Parameter(
        #     torch.from_numpy(np.random.randn(out_dim, in_dim)).type(torch.float32) / lr_mul
        # )  # 对齐paddle时测试权重

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        """
        forward func
        :param iuput: input
        """
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out


class GeneratorConv2d(nn.Module):
    """
    generator conv2d module reference paper
    """

    def __init__(self, in_channel, out_channel, kernel_size, upsample=False, downsample=False):
        """
        init
        :param in_channel: input channel
        :param out_channel: output channel
        :param kernel_size: conv kernel size
        :param upsample: whether use upsample
        :param downsample: whether use downsample
        """
        super(GeneratorConv2d, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = (kernel_size - 1) // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
            # torch.from_numpy(
            #     np.random.randn(1, out_channel, in_channel, kernel_size, kernel_size)
            # ).type(torch.float32)  # 对齐paddle时测试权重
        )

        if upsample:
            if kernel_size == 3:
                self.blur = Blur(stride=1, pad=(1, 1, 1, 1))
            elif kernel_size == 4:
                self.blur = Blur(stride=1, pad=(0, 1, 0, 1))

        if downsample:
            if kernel_size == 3:
                self.blur = Blur(stride=1, pad=(2, 2, 2, 2))
            elif kernel_size == 4:
                self.blur = Blur(stride=1, pad=(2, 3, 2, 3))

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        batch, in_channel, height, width = x.shape

        weight = self.weight * self.scale
        weight = weight.repeat(batch, 1, 1, 1, 1).view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            x = self.blur(x)
            _, _, height, width = x.shape
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class ToRGB(nn.Module):
    """
    to rgb module in generator reference paper
    """

    def __init__(self, in_channel, res_factor=2):
        """
        init
        :param in_channel: input channel
        :param res_factor: whether use upscale2d
        """
        super(ToRGB, self).__init__()

        self.conv = GeneratorConv2d(in_channel, 3, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        self.upscale2d = Upscale2d()
        self.blur = Blur(stride=1, pad=(1, 2, 1, 2))

        self.res_factor = res_factor

    def forward(self, x, skip=None):
        """
        forward func
        :param x: input
        :param skip: skip input
        """
        out = self.conv(x)
        out = out + self.bias

        if skip is not None:
            if self.res_factor == 2:
                skip = self.upscale2d(skip)
            skip = self.blur(skip)
            out = out + skip

        return out


class GeneratorDecoderBlock(nn.Module):
    """
    generator decoder block
    """

    def __init__(self, in_channel, out_channel, kernel_size, upsample=False):
        """
        init
        :param in_channel: input channel
        :param out_channel: output channel
        :param kernel_size: conv kernel size
        :param upsample: whether use upsample
        """
        super(GeneratorDecoderBlock, self).__init__()

        self.conv = GeneratorConv2d(in_channel, out_channel, kernel_size, upsample=upsample)
        # self.activate = nn.LeakyReLU(0.2)
        self.activate = nn.SiLU()

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.conv(x)
        out = self.activate(out)
        return out


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


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
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
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
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
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

    def forward(self, qkv):
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
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
            self,
            channels,
            num_heads=8,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
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
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class GeneratorDecoder(nn.Module):
    """
    generator decoder net for unet structure reference meitu: https://zhuanlan.zhihu.com/p/88442535
    """

    def __init__(self, size=512, channel_multiplier=2, generator_scale=1.0):
        """
        init
        :param size: generator input size
        :param channel_multiplier: channel multiplier factor
        """
        super(GeneratorDecoder, self).__init__()

        self.size = size
        self.channels = {
            4: int(512 * generator_scale),
            8: int(512 * generator_scale),
            16: int(512 * generator_scale),
            32: int(512 * generator_scale),
            64: int(256 * channel_multiplier * generator_scale),
            128: int(128 * channel_multiplier * generator_scale),
            256: int(64 * channel_multiplier * generator_scale),
            512: int(32 * channel_multiplier * generator_scale),
            1024: int(16 * channel_multiplier * generator_scale),
        }

        self.conv1 = GeneratorDecoderBlock(self.channels[4], self.channels[4], 3)
        self.to_rgb1 = ToRGB(self.channels[4])

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.attens = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            up_size = 2 ** i

            out_channel = self.channels[up_size]

            self.convs.append(GeneratorDecoderBlock(in_channel, out_channel, 3, upsample=True))
            self.convs.append(GeneratorDecoderBlock(out_channel, out_channel, 3))

            if up_size in [8, 16, 32, 64]:
                self.attens.append(AttentionBlock(out_channel))
            else:
                self.attens.append(Identity())

            self.to_rgbs.append(ToRGB(out_channel))

            in_channel = out_channel * 2

        out_channel = self.channels[2 ** self.log_size]
        self.conv2 = GeneratorDecoderBlock(in_channel, out_channel, 3)
        self.to_rgb2 = ToRGB(out_channel, res_factor=1)

    def make_noise(self, batch, device):
        """
        make noise input when only train decoder
        :param batch: batch size
        :param device: gpu device
        """
        connections = []
        for i in range(2, self.log_size + 1):
            channels = self.channels[2 ** i]

            connections.append(torch.randn(batch, channels, 2 ** i, 2 ** i).to(device))
        return connections[::-1]

    def unnormal_noise(self, feature, noise):
        """
        unnormal noise mean and std by decoder feature
        :param feature: decoder feature
        :param noise: input noise
        """
        in_mean, in_std = torch.mean(feature, dim=[2, 3], keepdim=True), torch.std(feature, dim=[2, 3], keepdim=True)

        batch, channel, _, _ = feature.shape

        random_choose = torch.randint(0, channel, (batch,)).to(feature.device)
        random_choose_index = torch.linspace(0, random_choose.shape[0] - 1,
                                             steps=random_choose.shape[0])
        random_choose_index = random_choose_index.type(torch.LongTensor).to(feature.device)
        choose = random_choose_index * channel + random_choose

        in_mean, in_std = in_mean.reshape(-1, 1, 1, 1), in_std.reshape(-1, 1, 1, 1)
        in_mean_choose, in_std_choose = in_mean[choose, :, :, :], in_std[choose, :, :, :]

        noise = noise * in_std_choose + in_mean_choose
        return noise

    def forward(self, connections, only_train_decoder=False):
        """
        forward func
        :param connections: connections feature by encoder or make_noise
        :param only_train_decoder: whether first train decoder, then train encoder
        """
        out = self.conv1(connections[-1])

        skip = self.to_rgb1(out)

        i = -2

        image_list = []
        for conv1, conv2, atten, to_rgb in zip(self.convs[::2], self.convs[1::2], self.attens, self.to_rgbs):
            out = conv1(out)
            out = conv2(out)

            skip = to_rgb(out, skip)

            # print(i, skip.shape)
            # 64, 128, 256
            if i in [-5, -6, -7]:
                image_list.append(skip)

            out = atten(out)
            noise = connections[i]
            out = torch.cat([out, noise], dim=1)

            i = i - 1

        out = self.conv2(out)
        skip = self.to_rgb2(out, skip)
        image_list.append(skip)

        return image_list


class GeneratorEncoderBlock(nn.Module):
    """
    generator encoder block
    """

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False):
        """
        init
        :param in_channel: input channel
        :param out_channel: ouptut channel
        :param kernel_size: conv kernel size
        :param downsample: whether use downsample
        """
        super(GeneratorEncoderBlock, self).__init__()

        self.conv = GeneratorConv2d(in_channel, out_channel, kernel_size, downsample=downsample)
        # self.activate = nn.LeakyReLU(0.2)
        self.activate = nn.SiLU()

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.conv(x)
        out = self.activate(out)
        return out


class GeneratorEncoder(nn.Module):
    """
    generator encoder net for unet structure reference meitu: https://zhuanlan.zhihu.com/p/88442535
    """

    def __init__(self, size=512, channel_multiplier=2, generator_scale=1.0):
        """
        init
        :param size: generator size
        :param channel_multiplier: channel multiplier factor
        """
        super(GeneratorEncoder, self).__init__()

        self.channels = {
            4: int(512 * generator_scale),
            8: int(512 * generator_scale),
            16: int(512 * generator_scale),
            32: int(512 * generator_scale),
            64: int(256 * channel_multiplier * generator_scale),
            128: int(128 * channel_multiplier * generator_scale),
            256: int(64 * channel_multiplier * generator_scale),
            512: int(32 * channel_multiplier * generator_scale),
            1024: int(16 * channel_multiplier * generator_scale),
        }

        self.conv1 = GeneratorEncoderBlock(3, self.channels[size], 3)

        log_size = int(math.log(size, 2))

        in_channel = self.channels[size]

        self.convs = nn.ModuleList()
        self.attens = nn.ModuleList()

        for i in range(log_size, 2, -1):
            down_size = 2 ** (i - 1)

            out_channel = self.channels[down_size]

            self.convs.append(GeneratorEncoderBlock(in_channel, out_channel, 3, downsample=True))
            self.convs.append(GeneratorEncoderBlock(out_channel, out_channel, 3))

            if down_size in [4, 8, 16, 32, 64]:
                self.attens.append(AttentionBlock(out_channel))
            else:
                self.attens.append(Identity())

            in_channel = out_channel

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        connections = []

        out = self.conv1(x)
        connections.append(out)

        for conv1, conv2, atten in zip(self.convs[::2], self.convs[1::2], self.attens):
            out = conv1(out)
            out = conv2(out)
            out = atten(out)
            connections.append(out)

        return connections


class Generator2D(nn.Module):
    """
    generator net
    """

    def __init__(self, size=512, channel_multiplier=2, generator_scale=1.0):
        """
        init
        :param size: input size
        :param channel_multiplier: channel multiplier factor
        """
        super(Generator2D, self).__init__()

        self.g_encoder = GeneratorEncoder(size, channel_multiplier, generator_scale)
        self.g_decoder = GeneratorDecoder(size, channel_multiplier, generator_scale)

    def forward(self, x):
        """
        init
        :param x: input
        """
        connections = self.g_encoder(x)
        out = self.g_decoder(connections)
        return out


if __name__ == '__main__':
    """
    python -m medical_projects.ct2cta.models.generator2d
    """
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    netG = Generator(in_size=512, model_type='stylegan2').to(device)
    count_parameters(netG)
    x = torch.randn((1, 3, 512, 512)).to(device)
    y_list = netG(x)
    for y in y_list:
        print(y.shape)