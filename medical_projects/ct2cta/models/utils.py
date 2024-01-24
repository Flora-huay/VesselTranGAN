# !/usr/bin/env python3
# coding=utf-8
import torch.nn as nn
import functools


def weights_init(m):
    """
    Args:
        m:
    Returns:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def count_parameters(model):
    """
    Args:
        model:
    Returns:
    """
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
