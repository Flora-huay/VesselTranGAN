# !/usr/bin/env python3
# coding=utf-8
import numpy as np

import torch


class PixelLoss(object):
    """
    pixel loss
    """
    def __init__(self):
        """
        init
        """
        pass

    def pixel_loss(self, source, target, loss_type="l1"):
        """
        pixel loss
        Args:
            source:
            target:
            loss_type:
        Returns:
        """
        assert source.shape == target.shape
        assert loss_type in ["l1", "l2"]

        diff = source - target

        # mask = (target >= 0.4).type(target.dtype)
        # mask = (0.2 + mask).to(target.device)
        mask1 = (target >= 0.4)
        mask2 = (0.6 >= target)
        mask = (mask1 * mask2).type(target.dtype).to(target.device)
        mask3 = (target > 0.6).type(target.dtype).to(target.device) * 0.5
        mask4 = (target < 0.4).type(target.dtype).to(target.device) * 0.5
        mask = mask + mask3 + mask4

        if loss_type == "l1":
            return torch.mean(torch.abs(diff * mask))
        elif loss_type == "l2":
            return torch.mean(torch.square(diff * mask))
        else:
            raise ValueError("not supported loss type: {}".format(loss_type))

    def eval_pixel_loss(self, source, target, loss_type="l1"):
        """
        Args:
            source:
            target:
            loss_type:
        Returns:
        """
        assert source.shape == target.shape
        assert loss_type in ["l1", "l2"]

        diff = source - target

        mask1 = (target >= 0.4)
        mask2 = (0.6 >= target)
        mask = (mask1 * mask2).type(target.dtype).to(target.device)

        all_dim = 1
        for dim in source.shape:
            all_dim *= dim
        scale = all_dim / (torch.sum(mask) + 1e-6)

        if loss_type == "l1":
            return torch.mean(torch.abs(diff * mask)) * scale
        elif loss_type == "l2":
            return torch.mean(torch.square(diff * mask)) * scale
        else:
            raise ValueError("not supported loss type: {}".format(loss_type))

    def eval_pixel_loss_numpy(self, source, target, loss_type="l1"):
        """
        Args:
            source:
            target:
            loss_type:
        Returns:
        """
        print(source.shape, target.shape)
        assert source.shape == target.shape
        assert loss_type in ["l1", "l2"]

        diff = source - target

        mask1 = (target >= 0.4)
        mask2 = (0.6 >= target)
        mask = (mask1 * mask2).astype(np.float64)

        all_dim = 1
        for dim in source.shape:
            all_dim *= dim
        scale = all_dim / (np.sum(mask) + 1e-6)

        if loss_type == "l1":
            return np.mean(np.abs(diff * mask)) * scale
        elif loss_type == "l2":
            return np.mean(np.square(diff * mask)) * scale
        else:
            raise ValueError("not supported loss type: {}".format(loss_type))
