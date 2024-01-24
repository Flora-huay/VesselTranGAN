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
    npy_root = "/lfs/huayun/npy_root"
    nii_save_root = "/lfs/huayun/npy_to_nii_root"
    create_dir(nii_save_root)

    npy_path_list = sorted(glob.glob(os.path.join(npy_root, '*.npy')))
    for npy_path in tqdm(npy_path_list):
        basename = os.path.splitext(os.path.basename(npy_path))[0]

        npy_data = np.load(npy_path) / 255.0

        nii_data = unnormalize_nii_data(npy_data)
        write_np_to_nii(nii_data, os.path.join(nii_save_root, '{}.nii.gz'.format(basename)))
