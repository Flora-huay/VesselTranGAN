# !/usr/bin/env python3
# coding=utf-8
#
"""
Authors: Zhang Shichang
"""
import torch
import torch.nn.functional as F


def vgg_loss(image_A, image_B, vgg_model):
    """
    only support torch1.7.1
    Vgg Loss, L2 norm.
    Args:
        imageA: torch.Tensor, (B, 3, H, W).
        image_B_features: torch.Tensor, (B, -1), the feature of target image_B.
        vgg_model: the vgg model.
    """
    if image_A.shape[2] > 256:
        image_A = F.interpolate(image_A, size=(256, 256), mode='area')
    if image_B.shape[2] > 256:
        image_B = F.interpolate(image_B, size=(256, 256), mode='area')
    image_A = image_A * 255.
    image_B = image_B * 255.
    image_A_features = vgg_model(image_A, resize_images=False, return_lpips=True)
    image_B_features = vgg_model(image_B, resize_images=False, return_lpips=True)
    dist = (image_A_features - image_B_features).square().sum()
    return dist
