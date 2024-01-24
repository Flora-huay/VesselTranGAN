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

    for person_root in tqdm(sorted(glob.glob(os.path.join(ori_datas_root, '*')))):
        if not os.path.isdir(person_root):
            continue

        date_root_list = sorted(glob.glob(os.path.join(person_root, '*')))
        date_root_list = [date_root for date_root in date_root_list if os.path.isdir(date_root)]
        if len(date_root_list) < 2:
            continue

        for date_root in date_root_list:
            ct_path = os.path.join(date_root, 'PS.nii.gz')
            cta_path = os.path.join(date_root, 'ZQ.nii.gz')

            if not os.path.exists(ct_path):
                print('ct data: {} is lossed!!!'.format(ct_path))
                continue
            if not os.path.exists(cta_path):
                print('cta data: {} is lossed!!!'.format(cta_path))
                continue

            basename = '{}_{}'.format(os.path.basename(person_root), os.path.basename(date_root))

            os.system('cp {} {}'.format(ct_path, os.path.join(test_ct_save_root, '{}.nii.gz'.format(basename))))
            os.system('cp {} {}'.format(cta_path, os.path.join(test_cta_save_root, '{}.nii.gz'.format(basename))))

            break
