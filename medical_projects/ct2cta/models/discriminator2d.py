# !/usr/bin/env python3
# coding=utf-8
#
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.cuda.amp import autocast, GradScaler

from .utils import get_norm_layer, weights_init, count_parameters
from .generator2d import EqualConv2d, Blur, EqualLinear


class Discriminator(nn.Module):
    """
    dis
    """
    def __init__(self, in_c=3, in_size=512,
                 model_type="stylegan2", use_amp=False, discriminator_scale=1.0):
        """
        init
        Args:
            in_size:
            model_type:
            use_amp:
        """
        super(Discriminator, self).__init__()

        self.model_type = model_type
        self.in_size = in_size
        self.use_amp = use_amp

        assert self.model_type in ["stylegan2"]
        assert self.in_size in [512]

        self.netD = Discriminator2D(
            in_c=in_c, size=in_size, channel_multiplier=1,
            discriminator_scale=discriminator_scale
        )

    def forward(self, x):
        """
        forward
        Args:
            x:
        Returns:
        """
        if self.use_amp:
            with autocast():
                return self.netD(x)
        else:
            return self.netD(x)


class DiscriminatorConvLayer(nn.Sequential):
    """
    discriminator conv module
    """

    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, activate=True, bias=True):
        """
        init
        :param in_channel: input channel
        :param out_channel: output channel
        :param kernel_size: conv kernel size
        :param downsample: whether use downsample
        :param activate: whether use activate func
        :param bias: whether use bias
        """
        layer = []

        if downsample:
            layer.append(Blur(stride=1, pad=(1, 1, 1, 1)))
            stride = 2
        else:
            stride = 1

        layer.append(
            EqualConv2d(in_channel, out_channel, kernel_size,
                        stride=stride, padding=(kernel_size - 1) // 2, bias=bias)
        )

        if activate:
            # layer.append(nn.LeakyReLU(0.2))
            layer.append(nn.SiLU())

        super(DiscriminatorConvLayer, self).__init__(*layer)


class DiscriminatorResblock(nn.Module):
    """
    discriminator resblock
    """

    def __init__(self, in_channel, out_channel):
        """
        init
        :param in_channel: input channel
        :param out_channel: output channel
        """
        super(DiscriminatorResblock, self).__init__()

        self.conv1 = DiscriminatorConvLayer(in_channel, out_channel, 3)
        self.conv2 = DiscriminatorConvLayer(out_channel, out_channel, 3, downsample=True)

        self.skip = DiscriminatorConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.conv1(x)
        out = self.conv2(out)

        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator2D(nn.Module):
    """
    discriminator net reference paper
    """

    def __init__(self, in_c=3, size=512, channel_multiplier=2, discriminator_scale=1.0):
        """
        init
        :param size: input size
        :param channel_multiplier: channel multiplier factor
        """
        super(Discriminator2D, self).__init__()

        channels = {
            4: int(512 * discriminator_scale),
            8: int(512 * discriminator_scale),
            16: int(512 * discriminator_scale),
            32: int(512 * discriminator_scale),
            64: int(256 * channel_multiplier * discriminator_scale),
            128: int(128 * channel_multiplier * discriminator_scale),
            256: int(64 * channel_multiplier * discriminator_scale),
            512: int(32 * channel_multiplier * discriminator_scale),
            1024: int(16 * channel_multiplier * discriminator_scale),
        }

        convs = [DiscriminatorConvLayer(in_c, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(DiscriminatorResblock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = DiscriminatorConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4]),
            # nn.LeakyReLU(0.2),
            nn.SiLU(),
            EqualLinear(channels[4], 1),
        )

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.convs(x)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


if __name__ == '__main__':
    """
    python -m medical_projects.ct2cta.models.discriminator2d
    """
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    netD = Discriminator(in_c=3 * 4, in_size=512, model_type="stylegan2").to(device)
    count_parameters(netD)
    x = torch.randn((1, 3 * 4, 512, 512)).to(device)
    y = netD(x)
    print(y.shape)
