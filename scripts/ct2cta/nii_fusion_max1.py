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
    path_2D = '/lfs/huayun/Fusion_modify/Output_2D/'
    path_3D = '/lfs/huayun/Fusion_modify/Output_3D/'
    save_root = '/lfs/huayun/Fusion_modify/fusion/'
    create_dir(save_root)

    cta_2d_paths = sorted(glob.glob(os.path.join(path_2D, '*.npy')))
    for cta_2d_path in tqdm(cta_2d_paths):
        basename = os.path.basename(cta_2d_path)

        cta_3d_path = os.path.join(path_3D, basename)
        if not os.path.exists(cta_3d_path):
            print('cta 3d: {} is lossed!!'.format(cta_3d_path))
            continue

        cta_2d_data = np.load(cta_2d_path)
        cta_3d_data = np.load(cta_3d_path)
        print(cta_2d_data.min(), cta_2d_data.max(), cta_3d_data.min(), cta_3d_data.max())

        if (cta_2d_data.shape[0] != cta_3d_data.shape[0]) or (cta_2d_data.shape[1] != cta_3d_data.shape[1]):
            print('shape error: 2d {}, 3d {}'.format(cta_2d_data.shape, cta_3d_data.shape))
            continue

        min_depth = min(cta_2d_data.shape[2], cta_3d_data.shape[2])
        cta_2d_data_norm = cta_2d_data[:, :, :min_depth] / 255.0
        cta_3d_data_norm = cta_3d_data[:, :, :min_depth] / 255.0

        cta_fusion_data_norm = np.maximum(cta_2d_data_norm, cta_3d_data_norm)

        cta_fusion_data = unnormalize_nii_data(cta_fusion_data_norm)
        basename = os.path.splitext(basename)[0] + ".nii.gz"
        write_np_to_nii(cta_fusion_data, os.path.join(save_root, basename))
