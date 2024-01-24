# !/usr/bin/env python3
# coding=utf-8
import os
import glob
import time
import random
import string


def get_random_code(ascii_length=30):
    """
    使用 随机ascii码 + us时间戳 方式组合随机码
    Returns:
        str
    """
    us_time = int(time.time() * 1000000)
    random_ascii = ''.join(random.sample(string.ascii_letters, ascii_length))
    return '{}_{}'.format(random_ascii, us_time)


def get_file_basename(path):
    """
    获得文件basename
    Args:
        path:

    Returns:

    """
    return os.path.splitext(os.path.basename(path))[0]


def create_dir(path):
    """
    创建文件
    Args:
        path
    Return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def _allocate_average(source_num: int,  pool_num: int):
    """
    把source_num数量均匀分配给pool_num个人
    Args:
        source_num:
        pool_num:
    Returns:
        source_num_per_pool: list. 每个人分配的数量

    """
    # mod, rem = divmod(file_num, pool_num)
    mod = source_num // pool_num
    rem = source_num % pool_num
    source_num_per_pool = [mod] * (pool_num - rem) + [mod + 1] * rem
    return source_num_per_pool


def allocate_device(start_idx, end_idx, pool_num, gpu_ids):
    """
    allocate pool and gpu_ids between start_idx and end_idx

    需要在每个gpu上分配连续的idx, 比如 start_idx = 0, end_idx = 8,
    gpu_ids = [0, 1, 2, 3]
    则
    gpu0 -> 0, 1
    gpu1 -> 2, 3
    gpu2 -> 4, 5
    gpu3 -> 6, 7

    Args:
        start_idx: start index
        end_idx: end index
        pool_num: pool num
        gpu_ids: [0, 1, 2, 3]
    Returns:
        start_list, end_list, new pool_num, new gpu_ids
        len(start_list) == len(end_list) == pool_num == len(gpu_ids)s
    """
    assert (len(gpu_ids) > 0), 'no gpu ids to use!!!!'

    if end_idx - start_idx < pool_num:
        start_list = list(range(start_idx, end_idx, 1))
        end_list = list(range(start_idx + 1, end_idx + 1, 1))
        pool_num = len(start_list)
    else:
        # 计算间隔列表
        step_list = []
        for i in range(pool_num):
            step_list.append(0)
        for i in range(end_idx - start_idx):
            step_index = i % pool_num
            step_list[step_index] += 1
        # step_list = _allocate_average(end_idx - start_idx, pool_num)

        start_list, end_list = [], []
        start, end = start_idx, start_idx + step_list[0]
        start_list.append(start)
        end_list.append(end)
        for i in range(1, pool_num):
            start, end = end, end + step_list[i]
            start_list.append(start)
            end_list.append(end)

    if len(gpu_ids) >= pool_num:
        gpu_ids = gpu_ids[:pool_num]
    else:
        while len(gpu_ids) <= pool_num:
            gpu_ids = gpu_ids + gpu_ids
        gpu_ids = gpu_ids[:pool_num]

    allocate_device = {"start_list": start_list,
                       "end_list": end_list,
                       "pool_num": pool_num,
                       "gpu_ids": gpu_ids}

    return allocate_device
