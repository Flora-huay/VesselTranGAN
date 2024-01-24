# !/usr/bin/env python3
# coding=utf-8
"""
opt:
    dataset_root:
    is_train:
    mean:
    batch_size:
    n_epochs:

    gpu_ids:
    use_ddp:
    ddp_port:
    world_size:
    serial_batches:
    num_threads:
"""
import os
import glob
from tqdm import tqdm
from abc import ABC, abstractmethod

import numpy as np
import cv2
import torch
import torch.utils.data as data

from ...utils.io_nii import normalize_nii_data


class BaseDataset(data.Dataset, ABC):
    """
    base dataset
    """
    def __init__(self, opt):
        """
        init
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.current_epoch = 0

    @abstractmethod
    def __len__(self):
        """
        len
        Returns:
        """
        return 0

    @abstractmethod
    def __getitem__(self, item):
        """
        get item
        """
        pass


class CT2CTADataset(BaseDataset, ABC):
    """
    ct2cta dataset
    """
    def __init__(self, opt):
        """
        init
        Args:
            opt:
        """
        super(CT2CTADataset, self).__init__(opt)

        self.ct_path_list, self.cta_path_list = make_dataset(self.opt.dataset_root)

        self.size = len(self.ct_path_list)

        self.name = 'train' if opt.is_train else 'val'

        self.train_size = int(self.size * 0.97)
        self.val_size = max(self.size - self.train_size, opt.batch_size)
        self.train_size = self.size - self.val_size
        print('train size: {}, val size: {}'.format(self.train_size, self.val_size))
        if self.opt.is_train:
            self.ct_path_list = self.ct_path_list[:self.train_size]
            self.cta_path_list = self.cta_path_list[:self.train_size]
        else:
            self.ct_path_list = self.ct_path_list[self.train_size:]
            self.cta_path_list = self.cta_path_list[self.train_size:]

        self.center_random_num = 1

    def __getitem__(self, item):
        """
        Args:
            item:
        Returns:
        """
        while 1:
            try:
                data = self.__my_getitem__(item)
                break
            except Exception as e:
                print(e)
                item = item + 1
        return data

    def __my_getitem__(self, item):
        """
        Args:
            item:
        Returns:
        """
        if self.opt.is_train:
            index = item % self.train_size
        else:
            index = item % self.val_size

        ct_path = self.ct_path_list[index]
        cta_path = self.cta_path_list[index]

        ct_data = np.load(ct_path)
        cta_data = np.load(cta_path)

        # [x, y, z], 0 - 1
        ct_data = normalize_nii_data(ct_data.astype(np.float32))
        cta_data = normalize_nii_data(cta_data.astype(np.float32))

        # [c, x, y, z], 0 - 1
        ct_data = torch.from_numpy(ct_data[None, :])
        cta_data = torch.from_numpy(cta_data[None, :])

        ct_data = ct_data - self.opt.mean
        cta_data = cta_data - self.opt.mean

        data = {
            "A": ct_data,
            "B": cta_data,
            "name": self.name
        }
        return data

    def __len__(self):
        """
        len
        Returns:
        """
        if self.opt.is_train:
            return self.train_size
        else:
            return self.val_size


class CT2CTACenterDataset(BaseDataset, ABC):
    """
    ct2cta dataset
    """
    def __init__(self, opt):
        """
        init
        Args:
            opt:
        """
        super(CT2CTACenterDataset, self).__init__(opt)

        self.ct_path_list, self.cta_path_list = make_dataset(self.opt.dataset_root)

        self.size = len(self.ct_path_list)

        self.name = 'train' if opt.is_train else 'val'

        self.train_size = int(self.size * 0.98)
        self.val_size = max(self.size - self.train_size, opt.batch_size)
        self.train_size = self.size - self.val_size
        if self.train_size < 5:
            self.train_size = self.size
            self.val_size = self.size
        print('train size: {}, val size: {}'.format(self.train_size, self.val_size))
        if self.opt.is_train:
            self.ct_path_list = self.ct_path_list[:self.train_size]
            self.cta_path_list = self.cta_path_list[:self.train_size]
        else:
            self.ct_path_list = self.ct_path_list[-self.val_size:]
            self.cta_path_list = self.cta_path_list[-self.val_size:]

        self.center_random_num = 4
        self.center_random_count = 0
        self.ct_data_cache = None
        self.cta_data_cache = None

        self.brain_sample_num_all = 4
        self.brain_sample_num = 3
        self.is_brain_index_list = []
        self.no_brain_index_list = []
        for i in range(len(self.ct_path_list)):
            if self.check_index(i):
                self.is_brain_index_list.append(i)
            else:
                self.no_brain_index_list.append(i)

    def __getitem__(self, item):
        """
        Args:
            item:
        Returns:
        """
        # while 1:
        #     try:
        #         data = self.__my_getitem__(item)
        #         break
        #     except Exception as e:
        #         print(e)
        #         item = item + 1
        data_path_index = item // self.center_random_num
        if data_path_index % self.brain_sample_num_all < self.brain_sample_num:
            while 1:
                try:
                    data_path_index = np.random.choice(self.is_brain_index_list, size=1)[0]
                    data = self.__my_getitem__(data_path_index * self.center_random_num)
                    if data is not None:
                        break
                except Exception as e:
                    print(e)
        else:
            while 1:
                try:
                    data_path_index = np.random.choice(self.no_brain_index_list, size=1)[0]
                    data = self.__my_getitem__(data_path_index * self.center_random_num)
                    if data is not None:
                        break
                except Exception as e:
                    print(e)
        return data

    def check_index(self, data_path_index):
        """
        Args:
            data_path_index:
        Returns:
        """
        ct_path = self.ct_path_list[data_path_index]
        cta_path = self.cta_path_list[data_path_index]
        # print('{} ct: {}, cta: {}'.format(self.name, ct_path, cta_path))

        basename = os.path.splitext(os.path.basename(cta_path))[0]
        index = int(basename.split("_")[-1])
        if index >= 230:
            is_brain = 1
        else:
            is_brain = 0
        # print('index: {}, is brain: {}'.format(index, is_brain))
        return is_brain

    def load_data(self, data_path_index):
        """
        Args:
            data_path_index:
        Returns:
        """
        ct_path = self.ct_path_list[data_path_index]
        cta_path = self.cta_path_list[data_path_index]
        # print('{} ct: {}, cta: {}'.format(self.name, ct_path, cta_path))

        ct_data = np.load(ct_path)
        cta_data = np.load(cta_path)

        # [x, y, z], 0 - 1
        self.ct_data_cache = normalize_nii_data(ct_data.astype(np.float32))
        self.cta_data_cache = normalize_nii_data(cta_data.astype(np.float32))

        assert min(self.ct_data_cache.shape[0], self.ct_data_cache.shape[1]) >= self.opt.center_in_size
        assert self.ct_data_cache.shape[2] >= self.opt.center_depth

    def __my_getitem__(self, item):
        """
        Args:
            item:
        Returns:
        """
        data_path_index = item // self.center_random_num

        is_brain = self.check_index(data_path_index)
        if self.center_random_count == 0:
            self.load_data(data_path_index)
            print('is brain: {}'.format(is_brain))

        self.center_random_count += 1
        if self.center_random_count == self.center_random_num:
            self.center_random_count = 0
            # print('start next data {}'.format(self.name))
        # print('{} / {}'.format(self.center_random_count, self.center_random_num))

        random_h = np.random.choice(
            list(range(0, self.ct_data_cache.shape[0] - self.opt.center_in_size)), 1
        )[0]
        random_w = np.random.choice(
            list(range(0, self.ct_data_cache.shape[1] - self.opt.center_in_size)), 1
        )[0]
        random_d = np.random.choice(
            list(range(0, self.ct_data_cache.shape[2] - self.opt.center_depth)), 1
        )[0]

        # [x, y, z], 0 - 1
        ct_data = self.ct_data_cache[
                  random_h: random_h + self.opt.center_in_size,
                  random_w: random_w + self.opt.center_in_size,
                  random_d: random_d + self.opt.center_depth
                  ]
        cta_data = self.cta_data_cache[
                   random_h: random_h + self.opt.center_in_size,
                   random_w: random_w + self.opt.center_in_size,
                   random_d: random_d + self.opt.center_depth
                   ]

        # [c, x, y, z], 0 - 1
        ct_data = torch.from_numpy(ct_data[None, :])
        cta_data = torch.from_numpy(cta_data[None, :])

        ct_data = ct_data - self.opt.mean
        cta_data = cta_data - self.opt.mean

        data = {
            "A": ct_data,
            "B": cta_data,
            "name": self.name,
            "is_brain": is_brain
        }
        return data

    def __len__(self):
        """
        len
        Returns:
        """
        if self.opt.is_train:
            return self.train_size * self.center_random_num
        else:
            return self.val_size * self.center_random_num


class CT2CTAImageDataset(BaseDataset, ABC):
    """
    ct2cta dataset
    """
    def __init__(self, opt):
        """
        init
        Args:
            opt:
        """
        super(CT2CTAImageDataset, self).__init__(opt)

        self.ct_path_list, self.cta_path_list = make_dataset(self.opt.dataset_root, file_format='png')

        self.size = len(self.ct_path_list)

        self.name = 'train' if opt.is_train else 'val'

        self.train_size = int(self.size * 0.98)
        self.val_size = max(self.size - self.train_size, opt.batch_size)
        self.train_size = self.size - self.val_size
        if self.train_size < 5:
            self.train_size = self.size
            self.val_size = self.size
        print('train size: {}, val size: {}'.format(self.train_size, self.val_size))
        if self.opt.is_train:
            self.ct_path_list = self.ct_path_list[:self.train_size]
            self.cta_path_list = self.cta_path_list[:self.train_size]
        else:
            self.ct_path_list = self.ct_path_list[-self.val_size:]
            self.cta_path_list = self.cta_path_list[-self.val_size:]

        self.brain_sample_num_all = 4
        self.brain_sample_num = 3
        self.is_brain_index_list = []
        self.no_brain_index_list = []
        for i in range(len(self.ct_path_list)):
            if self.check_index(i):
                self.is_brain_index_list.append(i)
            else:
                self.no_brain_index_list.append(i)

    def __getitem__(self, item):
        """
        Args:
            item:
        Returns:
        """
        data_path_index = item
        if data_path_index % self.brain_sample_num_all < self.brain_sample_num:
            while 1:
                try:
                    data_path_index = np.random.choice(self.is_brain_index_list, size=1)[0]
                    data = self.__my_getitem__(data_path_index)
                    if data is not None:
                        break
                except Exception as e:
                    print(e)
        else:
            while 1:
                try:
                    data_path_index = np.random.choice(self.no_brain_index_list, size=1)[0]
                    data = self.__my_getitem__(data_path_index)
                    if data is not None:
                        break
                except Exception as e:
                    print(e)
        return data

    def check_index(self, data_path_index):
        """
        Args:
            data_path_index:
        Returns:
        """
        ct_path = self.ct_path_list[data_path_index]
        cta_path = self.cta_path_list[data_path_index]
        # print('{} ct: {}, cta: {}'.format(self.name, ct_path, cta_path))

        basename = os.path.splitext(os.path.basename(cta_path))[0]
        index = int(basename.split("_")[-1])
        if index >= 230:
            is_brain = 1
        else:
            is_brain = 0
        # print('index: {}, is brain: {}'.format(index, is_brain))
        return is_brain

    def __my_getitem__(self, item):
        """
        Args:
            item:
        Returns:
        """
        data_path_index = item

        is_brain = self.check_index(data_path_index)

        ct_path = self.ct_path_list[data_path_index]
        cta_path = self.cta_path_list[data_path_index]

        ct_data = (cv2.imread(ct_path) / 255.0).astype(np.float32)
        cta_data = (cv2.imread(cta_path) / 255.0).astype(np.float32)

        # [c, x, y], 0 - 1
        ct_data = torch.from_numpy(ct_data.transpose((2, 0, 1)))
        cta_data = torch.from_numpy(cta_data.transpose((2, 0, 1)))

        ct_data = ct_data - self.opt.mean
        cta_data = cta_data - self.opt.mean

        data = {
            "A": ct_data,
            "B": cta_data,
            "name": self.name,
            "is_brain": is_brain
        }
        return data

    def __len__(self):
        """
        len
        Returns:
        """
        if self.opt.is_train:
            return self.train_size
        else:
            return self.val_size


class CustomCT2CTADataLoader(object):
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    def __init__(self, opt, rank=0):
        """
        init
        """
        self.opt = opt
        if self.opt.dataset_type == 'center':
            self.dataset = CT2CTACenterDataset(self.opt)
        if self.opt.dataset_type == 'image':
            self.dataset = CT2CTAImageDataset(self.opt)
        else:
            self.dataset = CT2CTACenterDataset(self.opt)
        self.sampler = None
        print("rank %d %s dataset [%s] was created" % (rank, self.dataset.name, type(self.dataset).__name__))
        if opt.use_ddp and opt.is_train:
            world_size = opt.world_size
            # self.sampler = torch.utils.data.distributed.DistributedSampler(
            #     self.dataset,
            #     num_replicas=world_size,
            #     rank=rank,
            #     shuffle=not opt.serial_batches
            # )
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=self.sampler,
                num_workers=int(opt.num_threads / world_size),
                batch_size=int(opt.batch_size / world_size),
                drop_last=True)
        else:
            # self.dataloader = torch.utils.data.DataLoader(
            #     self.dataset,
            #     batch_size=opt.batch_size,
            #     shuffle=(not opt.serial_batches) and opt.is_train,
            #     num_workers=int(opt.num_threads),
            #     drop_last=True
            # )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.num_threads),
                drop_last=True
            )

    def set_epoch(self, epoch):
        """

        Args:
            epoch:

        Returns:

        """
        self.dataset.current_epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def load_data(self):
        """

        Returns:

        """
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data


def make_dataset(dataset_root, file_format="npy"):
    """
    make dataset
    Args:
        dataset_root:
            *
                *
                    CT.nii.gz
                    CP.nii.gz
        file_format:
    Returns:
    """
    ct_paths = sorted(glob.glob(os.path.join(dataset_root, '*', 'ct', '*.{}'.format(file_format))))

    ct_path_list = []
    cta_path_list = []
    for ct_path in tqdm(ct_paths):
        file_basename = os.path.splitext(os.path.basename(ct_path))[0]
        dir_basename = os.path.basename(os.path.dirname(os.path.dirname(ct_path)))

        cta_path = os.path.join(dataset_root, dir_basename, 'cta', '{}.{}'.format(file_basename, file_format))
        assert os.path.exists(ct_path) and os.path.exists(cta_path)

        ct_path_list.append(ct_path)
        cta_path_list.append(cta_path)

    random_seed = 1234
    ct_path_list = np.random.RandomState(random_seed).choice(ct_path_list, size=len(ct_path_list), replace=False)
    cta_path_list = np.random.RandomState(random_seed).choice(cta_path_list, size=len(cta_path_list), replace=False)
    return ct_path_list, cta_path_list
