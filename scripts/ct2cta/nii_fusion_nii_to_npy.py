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
    save_nii_fig_single, unnormalize_nii_data, write_np_to_nii
from medical_projects.utils.io_file import create_dir


if __name__ == '__main__':
    path_2D = '/lfs/huayun/Fusion_modify/Output_2D/'
    path_3D = '/lfs/huayun/Fusion_modify/Output_3D/'

    cta_2d_paths = sorted(glob.glob(os.path.join(path_2D, '*.nii.gz')))
    for cta_2d_path in tqdm(cta_2d_paths):
        basename = os.path.basename(cta_2d_path)
        basename = os.path.splitext(basename)[0]
        basename = basename.replace('.nii', '')

        cta_2d_data = read_nii_to_np(cta_2d_path)

        cta_2d_data_norm = normalize_nii_data(cta_2d_data.astype(np.float32))

        cta_2d_data_norm = (cta_2d_data_norm * 255).astype(np.uint8)
        np.save(os.path.join(path_2D, '{}.npy'.format(basename)), cta_2d_data_norm)

    cta_3d_paths = sorted(glob.glob(os.path.join(path_3D, '*.nii.gz')))
    for cta_3d_path in tqdm(cta_3d_paths):
        basename = os.path.basename(cta_3d_path)
        basename = os.path.splitext(basename)[0]
        basename = basename.replace('.nii', '')

        cta_3d_data = read_nii_to_np(cta_3d_path)

        cta_3d_data_norm = normalize_nii_data(cta_3d_data.astype(np.float32))

        cta_3d_data_norm = (cta_3d_data_norm * 255).astype(np.uint8)
        np.save(os.path.join(path_3D, '{}.npy'.format(basename)), cta_3d_data_norm)
