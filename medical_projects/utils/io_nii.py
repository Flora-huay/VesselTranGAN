# !/usr/bin/env python3
# coding=utf-8
import os

import numpy as np
import cv2
import nibabel as nib

from .io_file import create_dir


def read_nii_to_np(nii_path):
    """
    read nii to numpy array
    Args:
        nii_path:
    Returns:
        numpy format
    """
    np_data = nib.load(nii_path).get_fdata()
    return np_data


def write_np_to_nii(np_data, nii_path, save_path):
    """
    Args:
        np_data:
        nii_path:
        save_path:
    Returns:
    """
    img = nib.load(nii_path)
    img_affine = img.affine
    nib.Nifti1Image(np_data, img_affine).to_filename(save_path)


def normalize_nii_data(nii_data, bound_min=-1024, bound_max=2048):
    """
    Args:
        nii_data: numpy format
        bound_min:
        bound_max:
    Returns:
        (0 - 1)
    """
    nii_data = nii_data.astype(np.float32)
    nii_data_norm = (nii_data - bound_min) / (bound_max - bound_min)
    nii_data_norm[nii_data_norm < 0.0] = 0.0
    nii_data_norm[nii_data_norm > 1.0] = 1.0
    return nii_data_norm


def unnormalize_nii_data(nii_data, bound_min=-1024, bound_max=2048):
    """
    Args:
        nii_data:
        bound_min:
        bound_max:
    Returns:
    """
    nii_data = nii_data * (bound_max - bound_min) + bound_min
    return nii_data


def center_crop(nii_data, center_size=432):
    """
    Args:
        nii_data:
        center_size:
    Returns:
    """
    h, w = center_size, center_size

    ori_h, ori_w, ori_c = nii_data.shape[0], nii_data.shape[1], nii_data.shape[2]
    new_h, new_w = min(ori_h, h), min(ori_w, w)
    center_h, center_w = ori_h // 2, ori_w // 2
    left_h, left_w = max(center_h - new_h // 2, 0), max(center_w - new_w // 2, 0)
    crop_nii_data = nii_data[left_h: left_h + new_h, left_w: left_w + new_w, :]  # 432, 432, ori_c
    return crop_nii_data


def center_crop_fill(full_data, crop_data):
    """
    Args:
        full_data:
        crop_data:
    Returns:
    """
    full_h, full_w, full_c = full_data.shape[0], full_data.shape[1], full_data.shape[2]
    crop_h, crop_w, crop_c = crop_data.shape[0], crop_data.shape[1], crop_data.shape[2]

    new_h, new_w = min(full_h, crop_h), min(full_w, crop_w)
    center_h, center_w = full_h // 2, full_w // 2
    left_h, left_w = max(center_h - new_h // 2, 0), max(center_w - new_w // 2, 0)
    full_data[left_h: left_h + new_h, left_w: left_w + new_w, :crop_c] = crop_data[:new_h, :new_w, :crop_c]
    return full_data


def save_nii_fig_single(nii_data, save_path):
    """
    Args:
        nii_data:
        save_path:
    Returns:
    """
    if len(nii_data.shape) == 2:
        pass
    else:
        raise ValueError("save nii fig single: not supported shape!")

    cv2.imwrite(save_path, (nii_data * 255.0).astype(np.uint8))


def split_ct_cta_to_patch(ct_data, cta_data, save_root, patch=128, stride=32):
    """
    Args:
        ct_data: (h, w, c)
        cta_data: (h, w, c)
        save_root:
        patch:
        stride:
    Returns:
    """
    create_dir(save_root)

    try:
        assert len(ct_data.shape) == 3
        assert ct_data.shape == cta_data.shape
        assert min(ct_data.shape) >= patch
        assert patch >= stride
    except Exception as e:
        print(e)
        return 0

    ct_save_root = os.path.join(save_root, 'ct')
    create_dir(ct_save_root)
    cta_save_root = os.path.join(save_root, 'cta')
    create_dir(cta_save_root)

    count = 0
    h, w, c = ct_data.shape

    # 正向采样
    for h_start in range(0, h - patch, stride):
        for w_start in range(0, w - patch, stride):
            for c_start in range(0, c - patch, stride):
                ct_data_patch = ct_data[h_start: h_start + patch, w_start: w_start + patch, c_start: c_start + patch]
                cta_data_patch = cta_data[h_start: h_start + patch, w_start: w_start + patch, c_start: c_start + patch]

                ct_save_path = os.path.join(ct_save_root, '{}_{}_{}.npy'.format(h_start, w_start, c_start))
                np.save(ct_save_path, ct_data_patch)
                cta_save_path = os.path.join(cta_save_root, '{}_{}_{}.npy'.format(h_start, w_start, c_start))
                np.save(cta_save_path, cta_data_patch)

                count += 1

    # 反向采样
    for h_start in range(h - patch, 0, -stride):
        for w_start in range(w - patch, 0, -stride):
            for c_start in range(c - patch, 0, -stride):
                ct_data_patch = ct_data[h_start: h_start + patch, w_start: w_start + patch, c_start: c_start + patch]
                cta_data_patch = cta_data[h_start: h_start + patch, w_start: w_start + patch, c_start: c_start + patch]

                ct_save_path = os.path.join(ct_save_root, '{}_{}_{}.npy'.format(h_start, w_start, c_start))
                np.save(ct_save_path, ct_data_patch)
                cta_save_path = os.path.join(cta_save_root, '{}_{}_{}.npy'.format(h_start, w_start, c_start))
                np.save(cta_save_path, cta_data_patch)

                count += 1

    return count


def split_ct_cta_to_patch_1(ct_data, cta_data, save_root, patch=128, stride=32):
    """
    Args:
        ct_data: (h, w, c)
        cta_data: (h, w, c)
        save_root:
        patch:
        stride:
    Returns:
    """
    create_dir(save_root)

    try:
        assert len(ct_data.shape) == 3
        assert ct_data.shape == cta_data.shape
        assert min(ct_data.shape) >= patch
        assert patch >= stride
    except Exception as e:
        print(e)
        return 0

    ct_save_root = os.path.join(save_root, 'ct')
    create_dir(ct_save_root)
    cta_save_root = os.path.join(save_root, 'cta')
    create_dir(cta_save_root)

    count = 0
    h, w, c = ct_data.shape

    # 正向采样
    for c_start in range(0, c - patch, stride):
        ct_data_patch = ct_data[:, :, c_start: c_start + patch]
        cta_data_patch = cta_data[:, :, c_start: c_start + patch]

        ct_save_path = os.path.join(ct_save_root, '{}_{}_{}.npy'.format(h, w, c_start))
        cta_save_path = os.path.join(cta_save_root, '{}_{}_{}.npy'.format(h, w, c_start))

        if os.path.exists(ct_save_path) and os.path.exists(cta_save_path):
            pass
        else:
            np.save(ct_save_path, ct_data_patch)
            np.save(cta_save_path, cta_data_patch)

        count += 1

    # 反向采样
    for c_start in range(c - patch, 0, -stride):
        ct_data_patch = ct_data[:, :, c_start: c_start + patch]
        cta_data_patch = cta_data[:, :, c_start: c_start + patch]

        ct_save_path = os.path.join(ct_save_root, '{}_{}_{}.npy'.format(h, w, c_start))
        cta_save_path = os.path.join(cta_save_root, '{}_{}_{}.npy'.format(h, w, c_start))

        if os.path.exists(ct_save_path) and os.path.exists(cta_save_path):
            pass
        else:
            np.save(ct_save_path, ct_data_patch)
            np.save(cta_save_path, cta_data_patch)

        count += 1

    return count
