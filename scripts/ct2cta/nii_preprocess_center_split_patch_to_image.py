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
    # datas_root = "/storage/huayun/dataset"
    # save_root = "/storage/huayun/dataset_split_image"
    datas_root = "/lfs/huayun/dataset"
    save_root = "/lfs/huayun/dataset_split_image"
    # datas_root = "/Users/zhangshichang/work/zsc_project/MedicalProjects_data/"
    # save_root = "/Users/zhangshichang/work/zsc_project/MedicalProjects_data/dataset_split_image"
    create_dir(save_root)

    h, w = 512, 512

    # ct_paths = sorted(glob.glob(os.path.join(datas_root, '*', '*', 'CT.nii.gz')))
    ct_paths = sorted(glob.glob(os.path.join(datas_root, '*', '*', 'PS.nii.gz')))

    for ct_path in tqdm(ct_paths):
        start_time = time.time()

        dir_name = os.path.dirname(ct_path)

        basename = '{}_{}'.format(
            os.path.basename(os.path.dirname(dir_name)), os.path.basename(dir_name)
        )
        save_path = os.path.join(save_root, basename)
        ct_save_path = os.path.join(save_path, 'ct')
        cta_save_path = os.path.join(save_path, 'cta')

        exists_ct_img_path_list = sorted(glob.glob(os.path.join(ct_save_path, '*.png')))
        exists_cta_img_path_list = sorted(glob.glob(os.path.join(cta_save_path, '*.png')))
        if len(exists_ct_img_path_list) >= 400 and len(exists_ct_img_path_list) == len(exists_cta_img_path_list):
            print('exists, continue')
            continue

        # cta_path = os.path.join(dir_name, 'CP.nii.gz')
        cta_path = os.path.join(dir_name, 'ZQ.nii.gz')
        print('start process: {}'.format(ct_path))

        ct_data = read_nii_to_np(ct_path)
        cta_data = read_nii_to_np(cta_path)

        # 节省4倍存储, 加快读取速度
        # ct_data = ct_data.astype(np.int16)
        # cta_data = cta_data.astype(np.int16)

        ori_h, ori_w, ori_c = ct_data.shape[0], ct_data.shape[1], ct_data.shape[2]
        new_h, new_w = min(ori_h, h), min(ori_w, w)

        center_h, center_w = ori_h // 2, ori_w // 2
        left_h, left_w = max(center_h - new_h // 2, 0), max(center_w - new_w // 2, 0)
        ct_data = ct_data[left_h: left_h + new_h, left_w: left_w + new_w, :]
        cta_data = cta_data[left_h: left_h + new_h, left_w: left_w + new_w, :]

        print('ct new h: {}, w: {}'.format(ct_data.shape[0], ct_data.shape[1]))
        print('cta new h: {}, w: {}'.format(cta_data.shape[0], cta_data.shape[1]))

        ct_data_norm = normalize_nii_data(ct_data.astype(np.float32))
        cta_data_norm = normalize_nii_data(cta_data.astype(np.float32))

        if ct_data_norm.shape != cta_data_norm.shape:
            continue

        create_dir(save_path)
        create_dir(ct_save_path)
        create_dir(cta_save_path)
        for d in tqdm(range(ct_data_norm.shape[2])):
            ct_data_ = ct_data_norm[:, :, d]
            cta_data_ = cta_data_norm[:, :, d]

            # np.save(os.path.join(ct_save_path, '{}_{}.npy'.format(basename, d)), ct_data_)
            # np.save(os.path.join(cta_save_path, '{}_{}.npy'.format(basename, d)), cta_data_)
            save_nii_fig_single(ct_data_, os.path.join(ct_save_path, '{}_{}.png'.format(basename, d)))
            save_nii_fig_single(cta_data_, os.path.join(cta_save_path, '{}_{}.png'.format(basename, d)))

        end_time = time.time()
        print("epnd time: {}".format((end_time - start_time) / (60 * 60)))
