# !/usr/bin/env python3
# coding=utf-8
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
from .generator import EqualConv3d, Blur3d, EqualLinear


class Discriminator(nn.Module):
    """
    dis
    """
    def __init__(self, input_nc=1, in_size=128, depth_size=128,
                 model_type="stylegan2", norm='batch', use_sigmoid=False, use_amp=False, discriminator_scale=1.0):
        """
        init
        Args:
            input_nc:
            in_size:
            model_type:
            norm:
            use_sigmoid:
            use_amp:
        """
        super(Discriminator, self).__init__()

        self.model_type = model_type
        self.in_size = in_size
        self.depth_size = depth_size
        self.norm = norm
        self.use_sigmoid = use_sigmoid
        self.use_amp = use_amp

        assert self.model_type in ["unet", "stylegan2"]
        assert self.in_size in [128, 256, 384]
        assert self.depth_size in [32, 64, 128]
        assert self.norm in ['batch', 'instance']

        norm_layer = get_norm_layer(norm_type=self.norm)

        self.netD = None
        if self.in_size == self.depth_size:
            if self.model_type == "unet":
                self.netD = NLayerDiscriminator(input_nc, ndf=64, n_layers=5,
                                                norm_layer=norm_layer, use_sigmoid=self.use_sigmoid)
                self.netD.apply(weights_init)
            elif self.model_type == "stylegan2":
                self.netD = StyleGAN2Discriminator(size=in_size, channel_multiplier=1)
            else:
                raise ValueError("Generator: not supported model type: {}".format(self.model_type))
        else:
            assert self.in_size == 384
            assert self.depth_size in [32, 64]
            assert self.model_type == "stylegan2"

            self.netD = StyleGAN2DiscriminatorCenter(
                size=in_size, depth_size=self.depth_size, channel_multiplier=1,
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


#######################################################################################################################
#######################################################################################################################
# stylegan2
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
            layer.append(Blur3d(stride=1, pad=(1, 1, 1, 1, 1, 1)))
            stride = 2
        else:
            stride = 1

        layer.append(
            EqualConv3d(in_channel, out_channel, kernel_size,
                        stride=stride, padding=(kernel_size - 1) // 2, bias=bias)
        )

        if activate:
            #layer.append(nn.LeakyReLU(0.2, True))
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


class StyleGAN2Discriminator(nn.Module):
    """
    discriminator net reference paper
    """
    def __init__(self, size=128, channel_multiplier=1):
        """
        init
        :param size: input size
        :param channel_multiplier: channel multiplier factor
        """
        super(StyleGAN2Discriminator, self).__init__()

        channels = {
            4: 512,
            8: 256,
            16: 256,
            32: 256,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier
        }
        # 备注基于 size=512
        convs = [DiscriminatorConvLayer(1, channels[size], 1)]

        log_size = int(math.log(size, 2))  # 9

        in_channel = channels[size]  # 64

        for i in range(log_size, 2, -1):
            # i = 9, 8, 7, 6, 5, 4, 3
            out_channel = channels[2 ** (i - 1)]  # 128, 256, 512, 512, 512, 512, 512

            convs.append(DiscriminatorResblock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = DiscriminatorConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4 * 4, channels[4]),
            # nn.LeakyReLU(0.2, True),
            nn.SiLU(),
            EqualLinear(channels[4], 1),
        )

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.convs(x)

        batch, channel, height, width, depth = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width, depth
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)  # 标准差 (-1, 1, c, h, w，d)
        stddev = stddev.mean([2, 3, 4, 5], keepdim=True).squeeze(2)  # (-1, 1, 1, 1, 1)
        stddev = stddev.repeat(group, 1, height, width, depth)  # (batch, 1, h, w)
        out = torch.cat([out, stddev], 1)  # (batch, 513, h, w)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


class StyleGAN2DiscriminatorCenter(nn.Module):
    """
    discriminator net reference paper
    """
    def __init__(self, size=384, depth_size=64, channel_multiplier=1, discriminator_scale=1.0):
        """
        init
        :param size: input size
        :param channel_multiplier: channel multiplier factor
        """
        super(StyleGAN2DiscriminatorCenter, self).__init__()

        assert depth_size in [32, 64]

        # channels = {
        #     12: 512,                        # d 2
        #     24: 256,                        # d 4
        #     48: 256,                        # d 8
        #     96: 256,                        # d 16
        #     192: 128 * channel_multiplier,  # d 32
        #     384: 64 * channel_multiplier    # d 64
        # }
        # channels = {
        #     12: 512,                        # d 2
        #     24: 256,                        # d 4
        #     48: 128,                        # d 8
        #     96: 128,                        # d 16
        #     192: 64 * channel_multiplier,  # d 32
        #     384: 32 * channel_multiplier    # d 64
        # }
        channels = {
            12: int(512 * discriminator_scale),                        # d 2
            24: int(256 * discriminator_scale),                        # d 4
            48: int(256 * discriminator_scale),                        # d 8
            96: int(128 * discriminator_scale),                        # d 16
            192: int(64 * channel_multiplier * discriminator_scale),  # d 32
            384: int(32 * channel_multiplier * discriminator_scale)    # d 64
        }
        # 备注基于 size=512
        convs = [DiscriminatorConvLayer(1, channels[size], 1)]

        log_size = int(math.log(128, 2))  # 9

        in_channel = channels[size]  # 64

        for i in range(log_size, 2, -1):
            # i = 9, 8, 7, 6, 5, 4, 3
            out_channel = channels[2 ** (i - 1) * 3]  # 128, 256, 512, 512, 512, 512, 512

            convs.append(DiscriminatorResblock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = DiscriminatorConvLayer(in_channel + 1, channels[12], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[12] * 12 * 12 * 2, channels[12]) if depth_size == 64 else
            EqualLinear(channels[12] * 12 * 12 * 1, channels[12]),
            # nn.LeakyReLU(0.2, True),
            nn.SiLU(),
            EqualLinear(channels[12], 1),
        )

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.convs(x)

        batch, channel, height, width, depth = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width, depth
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)  # 标准差 (-1, 1, c, h, w，d)
        stddev = stddev.mean([2, 3, 4, 5], keepdim=True).squeeze(2)  # (-1, 1, 1, 1, 1)
        stddev = stddev.repeat(group, 1, height, width, depth)  # (batch, 1, h, w)
        out = torch.cat([out, stddev], 1)  # (batch, 513, h, w)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
#######################################################################################################################
# unet
class NLayerDiscriminator(nn.Module):
    """
    unet dis
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        """
        init
        Args:
            input_nc:
            ndf:
            n_layers:
            norm_layer:
            use_sigmoid:
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        forward
        Args:
            input:
        Returns:
        """
        return self.model(input)
#######################################################################################################################
#######################################################################################################################


if __name__ == '__main__':
    """
    python -m medical_projects.ct2cta.models.discriminator
    """
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    netD = Discriminator(in_size=384, depth_size=64, model_type="stylegan2").to(device)
    count_parameters(netD)
    x = torch.randn((1, 1, 384, 384, 64)).to(device)
    y = netD(x)
    print(y.shape)
