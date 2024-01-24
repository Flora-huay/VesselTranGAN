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
from medical_projects.utils.io_nii import read_nii_to_np, normalize_nii_data, split_ct_cta_to_patch, save_nii_fig_single
from medical_projects.utils.io_file import create_dir


if __name__ == '__main__':
    datas_root = "./dataset"
    save_root = "./dataset_split"
    create_dir(save_root)

    patch = 128
    stride_list = [patch - 16, patch - 12, patch - 8, patch - 4]

    # ct_paths = sorted(glob.glob(os.path.join(datas_root, '*', '*', 'CT.nii.gz')))
    ct_paths = sorted(glob.glob(os.path.join(datas_root, '*', '*', 'PS.nii.gz')))

    for ct_path in tqdm(ct_paths):
        start_time = time.time()

        dir_name = os.path.dirname(ct_path)

        # cta_path = os.path.join(dir_name, 'CP.nii.gz')
        cta_path = os.path.join(dir_name, 'ZQ.nii.gz')
        print('start process: {}'.format(ct_path))

        ct_data = read_nii_to_np(ct_path)
        cta_data = read_nii_to_np(cta_path)

        # 节省4倍存储, 加快读取速度
        ct_data = ct_data.astype(np.int16)
        cta_data = cta_data.astype(np.int16)
        # 在dataloader中进行归一化, 加快读写
        # ct_data = normalize_nii_data(ct_data, bound_min=ct_bound_min, bound_max=ct_bound_max)
        # cta_data = normalize_nii_data(cta_data, bound_min=cta_bound_min, bound_max=cta_bound_max)

        basename = '{}_{}'.format(
            os.path.basename(os.path.dirname(dir_name)), os.path.basename(dir_name)
        )
        save_path = os.path.join(save_root, basename)
        create_dir(save_path)

        count_all = 0
        count = 0
        for stride in tqdm(stride_list):
            count = split_ct_cta_to_patch(ct_data, cta_data, save_path, patch=patch, stride=stride)
            count_all += count
        print('end process: {}'.format(ct_path))
        print('{} split num: {}'.format(ct_path, count_all))

        # 存一张数据用于可视化
        if count > 0:
            ct_data_visual = ct_data[:, :, ct_data.shape[2] // 2: ct_data.shape[2] // 2 + 1]
            cta_data_visual = cta_data[:, :, cta_data.shape[2] // 2: cta_data.shape[2] // 2 + 1]
            ct_data_visual = normalize_nii_data(ct_data_visual.astype(np.float32))
            cta_data_visual = normalize_nii_data(cta_data_visual.astype(np.float32))
            vis_data = np.concatenate([ct_data_visual, cta_data_visual], axis=1)[:, :, 0]
            save_nii_fig_single(vis_data, os.path.join(save_root, '{}.png'.format(basename)))

        end_time = time.time()
        print("epnd time: {}".format((end_time - start_time) / (60 * 60)))
