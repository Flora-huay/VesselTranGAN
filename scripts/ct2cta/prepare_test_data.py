# !/usr/bin/env python3
# coding=utf-8
import os
import sys
import glob
import time
from tqdm import tqdm

import numpy as np
import cv2

sys.path.append("../..")
from medical_projects.utils.io_nii import read_nii_to_np, normalize_nii_data, \
    save_nii_fig_single, split_ct_cta_to_patch_1
from medical_projects.utils.io_file import create_dir


if __name__ == '__main__':
    ori_datas_root = "/lfs/huayun/test_dataset"
    test_save_root = "/lfs/huayun/test_dataset_prepare"
    create_dir(test_save_root)
    test_ct_save_root = os.path.join(test_save_root, 'ct')
    create_dir(test_ct_save_root)
    test_cta_save_root = os.path.join(test_save_root, 'cta')
    create_dir(test_cta_save_root)

    # ct_paths = sorted(glob.glob(os.path.join(datas_root, '*', '*', 'CT.nii.gz')))
    ct_paths = sorted(glob.glob(os.path.join(datas_root, '*', '*', 'PS.nii.gz')))

    for ct_path in tqdm(ct_paths):
        start_time = time.time()

        dir_name = os.path.dirname(ct_path)

        basename = '{}_{}'.format(
            os.path.basename(os.path.dirname(dir_name)), os.path.basename(dir_name)
        )

        # cta_path = os.path.join(dir_name, 'CP.nii.gz')
        cta_path = os.path.join(dir_name, 'ZQ.nii.gz')

        if not os.path.exists(cta_path):
            print('cta data: {} is lossed!!!'.format(cta_path))
            continue

        os.system('cp {} {}'.format(ct_path, os.path.join(test_ct_save_root, '{}.nii.gz'.format(basename))))
        os.system('cp {} {}'.format(cta_path, os.path.join(test_cta_save_root, '{}.nii.gz'.format(basename))))
