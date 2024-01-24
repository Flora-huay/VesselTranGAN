# !/usr/bin/env python3
# coding=utf-8
import os
import sys
import glob
import time
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import nibabel as nib

sys.path.append("../..")
from medical_projects.utils.io_nii import read_nii_to_np, normalize_nii_data, \
    save_nii_fig_single, unnormalize_nii_data
from medical_projects.utils.io_file import create_dir


def write_np_to_nii(np_data, save_path):
    img_affine = np.array([
        [  -0.54296899,    0.        ,    0.        ,  139.        ],
        [   0.        ,   -0.54296899,    0.        ,  143.36199951],
        [   0.        ,    0.        ,    0.625     , -184.6499939 ],
        [   0.        ,    0.        ,    0.        ,    1.        ]
    ])
    nib.Nifti1Image(np_data, img_affine).to_filename(save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nii_cta_2d_root', type=str)
    parser.add_argument('--nii_cta_3d_root', type=str)
    parser.add_argument('--nii_cta_fusion_root', type=str)

    opt = parser.parse_args()

    create_dir(opt.nii_cta_fusion_root)

    cta_2d_paths = sorted(glob.glob(os.path.join(opt.nii_cta_2d_root, '*.npy')))
    for cta_2d_path in tqdm(cta_2d_paths):
        basename = os.path.basename(cta_2d_path)

        cta_3d_path = os.path.join(opt.nii_cta_3d_root, basename)
        if not os.path.exists(cta_3d_path):
            print('cta 3d: {} is lossed!!'.format(cta_3d_path))
            continue

        cta_2d_data = np.load(cta_2d_path)
        cta_3d_data = np.load(cta_3d_path)
        print(cta_2d_data.min(), cta_2d_data.max(), cta_3d_data.min(), cta_3d_data.max())

        if (cta_2d_data.shape[0] != cta_3d_data.shape[0]) or (cta_2d_data.shape[1] != cta_3d_data.shape[1]):
            print('shape error: 2d {}, 3d {}'.format(cta_2d_data.shape, cta_3d_data.shape))
            continue

        cta_2d_data_norm = cta_2d_data
        cta_3d_data_norm = cta_3d_data
        min_depth = min(cta_2d_data_norm.shape[2], cta_3d_data_norm.shape[2])

        cta_fusion_data_norm = []
        for d in tqdm(range(min_depth)):
            cta_2d_img = cta_2d_data_norm[:, :, d: d + 1]
            cta_3d_img = cta_3d_data_norm[:, :, d: d + 1]

            # 取最大值
            cta_fusion_img = np.maximum(cta_2d_img, cta_3d_img)
            cta_fusion_data_norm.append(cta_fusion_img)

        cta_fusion_data_norm = np.concatenate(cta_fusion_data_norm, axis=2) / 255.0

        cta_fusion_data = unnormalize_nii_data(cta_fusion_data_norm)
        basename = os.path.splitext(basename)[0] + ".nii.gz"
        write_np_to_nii(cta_fusion_data, os.path.join(opt.nii_cta_fusion_root, basename))
