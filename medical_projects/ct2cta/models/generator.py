# !/usr/bin/env python3
# coding=utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.cuda.amp import autocast, GradScaler

from .utils import get_norm_layer, weights_init, count_parameters


class Generator(nn.Module):
    """
    generator
    """
    def __init__(self, input_nc=1, output_nc=1, in_size=128, depth_size=128,
                 model_type="stylegan2", norm='batch', use_dropout=False, use_amp=False,
                 generator_scale=1.0, use_checkpoint=False):
        """
        init
        Args:
            input_nc:
            output_nc:
            in_size:
            depth_size:
            model_type:
            norm:
            use_dropout:
            use_amp:
            generator_scale:
            use_checkpoint:
        """
        super(Generator, self).__init__()

        self.model_type = model_type
        self.in_size = in_size
        self.depth_size = depth_size
        self.norm = norm
        self.use_dropout = use_dropout
        self.use_amp = use_amp
        
        assert self.model_type in ["unet", "stylegan2"]
        assert self.in_size in [128, 256, 384]
        assert self.depth_size in [32, 64, 128]
        assert self.norm in ['batch', 'instance']

        norm_layer = get_norm_layer(norm_type=self.norm)

        self.netG = None
        if self.in_size == self.depth_size:
            if self.model_type == "unet":
                if self.in_size == 128:
                    self.netG = UnetGenerator3d(input_nc, output_nc, 5, ngf=64,
                                                norm_layer=norm_layer, use_dropout=use_dropout)
                elif self.in_size == 256:
                    self.netG = UnetGenerator3d(input_nc, output_nc, 8, ngf=64,
                                                norm_layer=norm_layer, use_dropout=use_dropout)
                else:
                    raise ValueError("Generator: unet not supported in size: {}".format(self.in_size))

                self.netG.apply(weights_init)
            elif self.model_type == "stylegan2":
                if self.in_size == 128:
                    self.netG = StyleGAN2Generator(size=self.in_size, channel_multiplier=1)
                else:
                    raise ValueError("Generator: stylegan2 not supported in size: {}".format(self.in_size))
            else:
                raise ValueError("Generator: not supported model type: {}".format(self.model_type))
        else:
            assert self.in_size == 384
            assert self.depth_size in [32, 64]
            assert self.model_type == "stylegan2"

            self.netG = StyleGAN2GeneratorCenter(
                size=self.in_size, channel_multiplier=1, generator_scale=generator_scale, use_checkpoint=use_checkpoint)

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


#######################################################################################################################
#######################################################################################################################
# stylegan
class Blur3d(nn.Module):
    """
    blur ops follow official tensorflow code, but use depthwise conv
    """
    def __init__(self, pad, normalize=True, stride=1):
        """
        init
        :param normalize: whether normalize
        :param stride: blur stride
        :param pad: (left_w, right_w, top_h, bottom_h, depth_front, depth_back)
                    stride 1 pad (1, 1, 1, 1, 1, 1)
                    stride 2 pad (1, 2, 1, 2, 1, 2)
        """
        super(Blur3d, self).__init__()
        kernel = [1, 3, 3, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]  # 4 * 4 * 4 类似高斯核
        kernel = kernel[None, None]

        if normalize:
            kernel = kernel / kernel.sum()  # 1 x 1 x 4 x 4 x 4

        self.register_buffer('kernel', kernel)  # self.kernel 不会被更新
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        kernel = self.kernel.expand(x.size(1), -1, -1, -1, -1)
        x = F.pad(x, [self.pad[0], self.pad[1], self.pad[2], self.pad[3], self.pad[4], self.pad[5]])
        x = F.conv3d(
            x,
            kernel,
            stride=self.stride,
            groups=x.size(1)
        )
        return x


class Upscale3d(nn.Module):
    """
    upscale 2d follow official tensorflow code, like a strict nearest upsample
    """
    @staticmethod
    def upscale3d(x, factor=2, gain=1):
        """
        core func
        :param factor: upscale factor
        :param gain: gain for input
        """
        assert x.dim() == 5
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1, shape[4], 1).expand(
                -1, -1, -1, factor, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3], factor * shape[4])
        return x

    def __init__(self, factor=2, gain=1):
        """
        init
        :param factor: upscale factor
        :param gain: gain for input
        """
        super(Upscale3d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        return self.upscale3d(x, factor=self.factor, gain=self.gain)


class EqualConv3d(nn.Module):
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
        super(EqualConv3d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 3)

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
        out = F.conv3d(
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


class GeneratorConv3d(nn.Module):
    """
    generator conv2d module reference paper
    """
    def __init__(self, in_channel, out_channel, kernel_size, upsample=False, downsample=False):
        """
        init
        Args:
        in_channel: input channel (3)
        out_channel: output channel (64)
        kernel_size: conv kernel size (3)
        upsample: whether use upsample
        downsample: whether use downsample
        """
        super(GeneratorConv3d, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 3)
        self.padding = (kernel_size - 1) // 2
        # 手动初始化，卷积层参数
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size, kernel_size)
        )

        if upsample:
            if kernel_size == 3:
                self.blur = Blur3d(stride=1, pad=(1, 1, 1, 1, 1, 1))
            elif kernel_size == 4:
                self.blur = Blur3d(stride=1, pad=(0, 1, 0, 1, 0, 1))

        if downsample:
            if kernel_size == 3:
                self.blur = Blur3d(stride=1, pad=(2, 2, 2, 2, 2, 2))
            elif kernel_size == 4:
                self.blur = Blur3d(stride=1, pad=(2, 3, 2, 3, 2, 3))

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        batch, in_channel, height, width, depth = x.shape

        weight = self.weight * self.scale
        # 前向时，把batch维度放到channel维度，并用group卷积。本质上与普通卷积一样，可能可以加速。
        # 故这里对权重进行repeat操作
        weight = weight.repeat(batch, 1, 1, 1, 1, 1).view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            x = x.view(1, batch * in_channel, height, width, depth)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose3d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width, depth = out.shape
            out = out.view(batch, self.out_channel, height, width, depth)
            out = self.blur(out)

        elif self.downsample:
            x = self.blur(x)
            _, _, height, width, depth = x.shape
            x = x.view(1, batch * in_channel, height, width, depth)
            out = F.conv3d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width, depth = out.shape
            out = out.view(batch, self.out_channel, height, width, depth)

        else:
            x = x.view(1, batch * in_channel, height, width, depth)
            out = F.conv3d(x, weight, padding=self.padding, groups=batch)
            _, _, height, width, depth = out.shape
            out = out.view(batch, self.out_channel, height, width, depth)

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
        # 1 x 1卷积压缩channal维度为1，语义上对应医学图像channel
        self.conv = GeneratorConv3d(in_channel, 1, 1)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

        self.upscale3d = Upscale3d()
        self.blur = Blur3d(stride=1, pad=(1, 2, 1, 2, 1, 2))

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
                # 对来自上一层对低清特征skip，进行上采样upscale
                skip = self.upscale3d(skip)
            # 高斯模糊
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

        self.conv = GeneratorConv3d(in_channel, out_channel, kernel_size, upsample=upsample)
        # self.activate = nn.LeakyReLU(0.2, True)
        self.activate = nn.SiLU()

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.conv(x)
        out = self.activate(out)
        return out


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

        self.conv = GeneratorConv3d(in_channel, out_channel, kernel_size, downsample=downsample)
        # self.activate = nn.LeakyReLU(0.2, True)
        self.activate = nn.SiLU()

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        out = self.conv(x)
        out = self.activate(out)
        return out


class GeneratorDecoder(nn.Module):
    """
    generator decoder net for unet structure reference meitu: https://zhuanlan.zhihu.com/p/88442535
    """
    def __init__(self, size=128, channel_multiplier=1):
        """
        init
        :param size: generator input size
        :param channel_multiplier: channel multiplier factor
        """
        super(GeneratorDecoder, self).__init__()

        self.size = size  # 512
        self.channels = {
            4: 512,
            8: 256,
            16: 256,
            32: 256,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier
        }

        self.conv1 = GeneratorDecoderBlock(self.channels[4], self.channels[4], 3)
        self.to_rgb1 = ToRGB(self.channels[4])

        self.log_size = int(math.log(size, 2))  # 9

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]  # 512

        for i in range(3, self.log_size + 1):
            # i = 3, 4, 5, 6, 7
            out_channel = self.channels[2 ** i]  # 256, 256, 256, 128, 64

            self.convs.append(GeneratorDecoderBlock(in_channel, out_channel, 3, upsample=True))
            self.convs.append(GeneratorDecoderBlock(out_channel, out_channel, 3))

            self.to_rgbs.append(ToRGB(out_channel))

            in_channel = out_channel * 2

        out_channel = self.channels[2 ** self.log_size]  # 64
        self.conv2 = GeneratorDecoderBlock(in_channel, out_channel, 3)
        self.to_rgb2 = ToRGB(out_channel, res_factor=1)

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
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out)
            out = conv2(out)
            skip = to_rgb(out, skip)

            # 16, 32, 64
            if i in [-3, -4, -5]:
                image_list.append(skip)

            noise = connections[i]
            out = torch.cat([out, noise], dim=1)

            i = i - 1

        out = self.conv2(out)
        skip = self.to_rgb2(out, skip)
        image_list.append(skip)

        return image_list


class GeneratorEncoder(nn.Module):
    """
    generator encoder net for unet structure reference meitu: https://zhuanlan.zhihu.com/p/88442535
    """
    def __init__(self, size=128, channel_multiplier=1):
        """
        init
        :param size: generator size
        :param channel_multiplier: channel multiplier factor
        """
        super(GeneratorEncoder, self).__init__()

        self.channels = {
            4: 512,
            8: 256,
            16: 256,
            32: 256,
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier
        }

        self.conv1 = GeneratorEncoderBlock(1, self.channels[size], 3)

        # 备注基于size为512
        log_size = int(math.log(size, 2))  # 9

        in_channel = self.channels[size]  # 64
        self.convs = nn.ModuleList()

        for i in range(log_size, 2, -1):
            # i = 9, 8, 7, 6, 5, 4, 3
            out_channel = self.channels[2 ** (i - 1)]  # 128, 256, 512, 512, 512, 512, 512

            self.convs.append(GeneratorEncoderBlock(in_channel, out_channel, 3, downsample=True))
            self.convs.append(GeneratorEncoderBlock(out_channel, out_channel, 3))

            in_channel = out_channel

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        connections = []

        out = self.conv1(x)
        connections.append(out)

        for conv1, conv2 in zip(self.convs[::2], self.convs[1::2]):
            out = conv1(out)
            out = conv2(out)
            connections.append(out)

        return connections


class StyleGAN2Generator(nn.Module):
    """
    generator net
    """
    def __init__(self, size=128, channel_multiplier=1):
        """
        init
        :param size: input size
        :param channel_multiplier: channel multiplier factor
        """
        super(StyleGAN2Generator, self).__init__()

        self.g_encoder = GeneratorEncoder(size, channel_multiplier)
        self.g_decoder = GeneratorDecoder(size, channel_multiplier)

    def forward(self, x):
        """
        init
        :param x: input
        """
        connections = self.g_encoder(x)
        out = self.g_decoder(connections)
        return out


class GeneratorDecoderCenter(nn.Module):
    """
    generator decoder net for unet structure reference meitu: https://zhuanlan.zhihu.com/p/88442535
    """
    def __init__(self, size=384, channel_multiplier=1, generator_scale=1.0, use_checkpoint=False):
        """
        init
        :param size: generator input size
        :param channel_multiplier: channel multiplier factor
        """
        super(GeneratorDecoderCenter, self).__init__()

        self.use_checkpoint = use_checkpoint

        self.size = size
        # self.channels = {
        #     12: 512,                        # d 2
        #     24: 256,                        # d 4
        #     48: 256,                        # d 8
        #     96: 256,                        # d 16
        #     192: 128 * channel_multiplier,  # d 32
        #     384: 64 * channel_multiplier    # d 64
        # }
        # self.channels = {
        #     12: 512,                        # d 2
        #     24: 256,                        # d 4
        #     48: 128,                        # d 8
        #     96: 128,                        # d 16
        #     192: 64 * channel_multiplier,  # d 32
        #     384: 32 * channel_multiplier    # d 64
        # }
        self.channels = {
            12: int(512 * generator_scale),                        # d 2
            24: int(256 * generator_scale),                        # d 4
            48: int(256 * generator_scale),                        # d 8
            96: int(128 * generator_scale),                        # d 16
            192: int(64 * channel_multiplier * generator_scale),  # d 32
            384: int(32 * channel_multiplier * generator_scale)    # d 64
        }

        self.conv1 = GeneratorDecoderBlock(self.channels[12], self.channels[12], 3)
        self.to_rgb1 = ToRGB(self.channels[12])

        self.log_size = int(math.log(128, 2))  # 9

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[12]  # 512

        for i in range(3, self.log_size + 1):
            # i = 3, 4, 5, 6, 7
            out_channel = self.channels[2 ** i * 3]  # 256, 256, 256, 128, 64

            self.convs.append(GeneratorDecoderBlock(in_channel, out_channel, 3, upsample=True))
            self.convs.append(GeneratorDecoderBlock(out_channel, out_channel, 3))

            self.to_rgbs.append(ToRGB(out_channel))

            in_channel = out_channel * 2

        out_channel = self.channels[self.size]  # 64
        self.conv2 = GeneratorDecoderBlock(in_channel, out_channel, 3)
        self.to_rgb2 = ToRGB(out_channel, res_factor=1)

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
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            if not self.use_checkpoint:
                out = conv1(out)
                out = conv2(out)
            else:
                out = torch.utils.checkpoint.checkpoint(conv1, out)
                out = torch.utils.checkpoint.checkpoint(conv2, out)
            skip = to_rgb(out, skip)

            # 16, 32, 64
            if i in [-3, -4, -5]:
                image_list.append(skip)

            noise = connections[i]
            out = torch.cat([out, noise], dim=1)

            i = i - 1

        out = self.conv2(out)
        skip = self.to_rgb2(out, skip)
        image_list.append(skip)

        return image_list


class GeneratorEncoderCenter(nn.Module):
    """
    generator encoder net for unet structure reference meitu: https://zhuanlan.zhihu.com/p/88442535
    """
    def __init__(self, size=384, channel_multiplier=1, generator_scale=1.0, use_checkpoint=False):
        """
        init
        :param size: generator size
        :param channel_multiplier: channel multiplier factor
        """
        super(GeneratorEncoderCenter, self).__init__()

        self.use_checkpoint = use_checkpoint
        # self.channels = {
        #     12: 512,                        # d 2
        #     24: 256,                        # d 4
        #     48: 256,                        # d 8
        #     96: 256,                        # d 16
        #     192: 128 * channel_multiplier,  # d 32
        #     384: 64 * channel_multiplier    # d 64
        # }
        # self.channels = {
        #     12: 512,                        # d 2
        #     24: 256,                        # d 4
        #     48: 128,                        # d 8
        #     96: 128,                        # d 16
        #     192: 64 * channel_multiplier,  # d 32
        #     384: 32 * channel_multiplier    # d 64
        # }
        self.channels = {
            12: int(512 * generator_scale),                        # d 2
            24: int(256 * generator_scale),                        # d 4
            48: int(256 * generator_scale),                        # d 8
            96: int(128 * generator_scale),                        # d 16
            192: int(64 * channel_multiplier * generator_scale),  # d 32
            384: int(32 * channel_multiplier * generator_scale)    # d 64
        }

        self.conv1 = GeneratorEncoderBlock(1, self.channels[size], 3)

        log_size = int(math.log(128, 2))  # 9

        in_channel = self.channels[size]  # 64
        self.convs = nn.ModuleList()

        for i in range(log_size, 2, -1):
            # i = 9, 8, 7, 6, 5, 4, 3
            out_channel = self.channels[2 ** (i - 1) * 3]  # 128, 256, 512, 512, 512, 512, 512

            self.convs.append(GeneratorEncoderBlock(in_channel, out_channel, 3, downsample=True))
            self.convs.append(GeneratorEncoderBlock(out_channel, out_channel, 3))

            in_channel = out_channel

    def forward(self, x):
        """
        forward func
        :param x: input
        """
        connections = []

        out = self.conv1(x)
        connections.append(out)

        for conv1, conv2 in zip(self.convs[::2], self.convs[1::2]):
            if not self.use_checkpoint:
                out = conv1(out)
                out = conv2(out)
            else:
                out = torch.utils.checkpoint.checkpoint(conv1, out)
                out = torch.utils.checkpoint.checkpoint(conv2, out)
            connections.append(out)

        return connections


class StyleGAN2GeneratorCenter(nn.Module):
    """
    generator net
    """
    def __init__(self, size=384, channel_multiplier=1, generator_scale=1.0, use_checkpoint=False):
        """
        init
        :param size: input size
        :param channel_multiplier: channel multiplier factor
        """
        super(StyleGAN2GeneratorCenter, self).__init__()

        self.g_encoder = GeneratorEncoderCenter(
            size, channel_multiplier, generator_scale=generator_scale, use_checkpoint=use_checkpoint)
        self.g_decoder = GeneratorDecoderCenter(
            size, channel_multiplier, generator_scale=generator_scale, use_checkpoint=use_checkpoint)

    def forward(self, x):
        """
        init
        :param x: input
        """
        connections = self.g_encoder(x)
        out = self.g_decoder(connections)
        return out
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
#######################################################################################################################
# unet
class UnetGenerator3d(nn.Module):
    """
    generator
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        """
        init
        Args:
            input_nc:
            output_nc:
            num_downs:
            ngf:
            norm_layer:
            use_dropout:
        """
        super(UnetGenerator3d, self).__init__()

        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                   use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3d(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        return [output]


class UnetSkipConnectionBlock3d(nn.Module):
    """
    unet skip connection block3d
    """
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        """
        init
        Args:
            outer_nc:
            inner_nc:
            submodule:
            outermost:
            innermost:
            norm_layer:
            use_dropout:
        """
        super(UnetSkipConnectionBlock3d, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)
#######################################################################################################################
#######################################################################################################################


if __name__ == '__main__':
    """
    python -m medical_projects.ct2cta.models.generator
    """
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    netG = Generator(1, 1, in_size=384, depth_size=32, model_type='stylegan2').to(device)
    count_parameters(netG)
    x = torch.randn((1, 1, 384, 384, 32)).to(device)
    y_list = netG(x)
    for y in y_list:
        print(y.shape)
