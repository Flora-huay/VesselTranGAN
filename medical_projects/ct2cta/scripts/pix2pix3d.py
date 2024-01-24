# !/usr/bin/env python3
# coding=utf-8
"""
opt:
    dataset_root:
    is_train:
    mean:
    batch_size:
    n_epochs:

    in_size:
    model_type:
    vgg16_model_path:
    lr_g:
    lr_d:
    beta1:
    beta2:
    display_batch:
    save_path:
    lambda_pixel:
    lambda_vgg:

    gpu_ids:
    use_ddp:
    ddp_port:
    world_size:
    serial_batches:
    num_threads:
"""
import os
import sys
import glob
import time
from tqdm import tqdm

import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from ..loss.gan_loss import LSGAN
from ..loss.pixel_loss import PixelLoss
from ..models.generator import Generator
from ..models.discriminator import Discriminator
from ..datasets.dataset import CustomCT2CTADataLoader
from ...utils.io_nii import read_nii_to_np, normalize_nii_data, unnormalize_nii_data, \
    center_crop, center_crop_fill, write_np_to_nii


def requires_grad(model, flag=True):
    """
    set requires grad flag
    :param model: model class
    :param flag: grad flag
    """
    for p in model.parameters():
        p.requires_grad = flag


class Pix2Pix3d(object):
    """
    pix2pix3d
    """
    def __init__(self, opt):
        """
        init
        Args:
            opt:
        """
        self.opt = opt

        if self.opt.use_gan_loss:
            self.model_names = ["netG", "netD"]
        else:
            self.model_names = ["netG"]
        self.parallel_names = self.model_names

        self.device = torch.device('cpu')

        if self.opt.dataset_type == 'center':
            self.netG = Generator(
                in_size=self.opt.center_in_size, depth_size=self.opt.center_depth,
                model_type=self.opt.model_type, use_amp=self.opt.use_amp,
                generator_scale=self.opt.generator_scale, use_checkpoint=self.opt.use_checkpoint
            )
            if self.opt.use_gan_loss:
                self.netD = Discriminator(
                    in_size=self.opt.center_in_size, depth_size=self.opt.center_depth,
                    model_type=self.opt.model_type, use_amp=self.opt.use_amp,
                    discriminator_scale=self.opt.discriminator_scale
                )
        else:
            self.netG = Generator(
                in_size=self.opt.in_size, depth_size=self.opt.in_size,
                model_type=self.opt.model_type, use_amp=self.opt.use_amp
            )
            if self.opt.use_gan_loss:
                self.netD = Discriminator(
                    in_size=self.opt.in_size, depth_size=self.opt.in_size,
                    model_type=self.opt.model_type, use_amp=self.opt.use_amp,
                    discriminator_scale=self.opt.discriminator_scale
                )

        self.netG.train()
        if self.opt.use_gan_loss:
            self.netD.train()

        self.gan_loss = LSGAN()
        self.pixel_loss = PixelLoss()

        if self.opt.use_adamw:
            self.optimizerG = optim.AdamW(self.netG.parameters(), lr=self.opt.lr_g,
                                          betas=(self.opt.beta1, self.opt.beta2), weight_decay=0.0)
            if self.opt.use_gan_loss:
                self.optimizerD = optim.AdamW(self.netD.parameters(), lr=self.opt.lr_d,
                                              betas=(self.opt.beta1, self.opt.beta2), weight_decay=0.0)
        else:
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr_g,
                                         betas=(self.opt.beta1, self.opt.beta2), weight_decay=0.0)
            if self.opt.use_gan_loss:
                self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr_d,
                                             betas=(self.opt.beta1, self.opt.beta2), weight_decay=0.0)

        self.ganIterations = 0
        self.iterations_list = []
        self.g_loss_list = []
        self.d_loss_list = []
        self.g_img_list = []
        self.g_vgg_list = []
        self.epoch_list = []
        self.val_img_list = []
        self.is_brain_val_img_list = []
        self.no_brain_val_img_list = []

        self.g_loss_list_path = os.path.join(self.opt.save_path, 'loss_g.npy')
        if os.path.exists(self.g_loss_list_path):
            self.g_loss_list = np.load(self.g_loss_list_path).tolist()
        self.d_loss_list_path = os.path.join(self.opt.save_path, 'loss_d.npy')
        if os.path.exists(self.d_loss_list_path):
            self.d_loss_list = np.load(self.d_loss_list_path).tolist()
        self.g_img_list_path = os.path.join(self.opt.save_path, 'loss_img.npy')
        if os.path.exists(self.g_img_list_path):
            self.g_img_list = np.load(self.g_img_list_path).tolist()
        self.val_img_list_path = os.path.join(self.opt.save_path, 'loss_val_img.npy')
        if os.path.exists(self.val_img_list_path):
            self.val_img_list = np.load(self.val_img_list_path).tolist()
        self.is_brain_val_img_list_path = os.path.join(self.opt.save_path, 'loss_val_img_is_brain.npy')
        if os.path.exists(self.is_brain_val_img_list_path):
            self.is_brain_val_img_list = np.load(self.is_brain_val_img_list_path).tolist()
        self.no_brain_val_img_list_path = os.path.join(self.opt.save_path, 'loss_val_img_no_brain.npy')
        if os.path.exists(self.no_brain_val_img_list_path):
            self.no_brain_val_img_list = np.load(self.no_brain_val_img_list_path).tolist()

        self.ganIterations = len(self.g_loss_list)
        self.iterations_list = list(range(0, self.ganIterations))
        self.epoch_list = list(range(0, len(self.val_img_list)))

    def parallelize(self, convert_sync_batchnorm=True):
        """
        Args:
            convert_sync_batchnorm:
        Returns:
        """
        if not self.opt.use_ddp:
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if module is not None:
                        setattr(self, name, module.to(self.device))
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if module is not None:
                        if convert_sync_batchnorm:
                            module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                        setattr(self, name, torch.nn.parallel.DistributedDataParallel(
                            module.to(self.device),
                            device_ids=[self.device.index],
                            find_unused_parameters=False, broadcast_buffers=False))

            # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    if module is not None:
                        setattr(self, name, module.to(self.device))

    def vis_train_info(self, epoch, iter, all_iter, loss_info):
        """
        Args:
            epoch:
            iter:
            loss_info:
        Returns:
        """
        info = 'epoch:[{}/{}], iter:[{}/{}]'.format(epoch, self.opt.n_epochs, iter, all_iter)
        for key in loss_info.keys():
            info += ', {}:{}'.format(key, loss_info[key])
        print(info)

    def mean_filter(self, lmks_list, weight=0.8, kernel=3):
        """
        mean filter for landmarks
        Args:
            landmarks list
        Return:
            post process landmarks list
        """
        if kernel == 3:
            new_lmks_list = [lmks_list[0]]

            for i in range(1, len(lmks_list) - 1):
                new_lmks = lmks_list[i - 1] * (1 - weight) / 2.0 + \
                           lmks_list[i] * weight + lmks_list[i + 1] * (1 - weight) / 2.0
                new_lmks_list.append(new_lmks)
            new_lmks_list.append(lmks_list[-1])
        elif kernel == 5:
            new_lmks_list = [lmks_list[0], lmks_list[1]]

            for i in range(2, len(lmks_list) - 2):
                new_lmks = lmks_list[i - 2] * (1 - weight) / 6.0 + lmks_list[i - 1] * (1 - weight) / 3.0 + \
                           lmks_list[i] * weight + lmks_list[i + 1] * (1 - weight) / 3.0 + \
                           lmks_list[i + 2] * (1 - weight) / 6.0

            new_lmks_list.extend([lmks_list[-2], lmks_list[-1]])

        return new_lmks_list

    def write_train_info(self, d_loss, g_loss, L_img, errVGG):
        """
        Args:
            d_loss:
            g_loss:
            L_img:
            errVGG:
            epoch:
        Returns:
        """
        self.iterations_list.append(self.ganIterations)
        self.g_loss_list.append(g_loss)
        self.d_loss_list.append(d_loss)
        self.g_img_list.append(L_img)

        smooth_g_loss_list = self.mean_filter(self.g_loss_list, weight=0.6, kernel=3)
        smooth_d_loss_list = self.mean_filter(self.d_loss_list, weight=0.6, kernel=3)
        smooth_g_img_list = self.mean_filter(self.g_img_list, weight=0.6, kernel=3)
        smooth_val_img_list = self.mean_filter(self.val_img_list, weight=0.6, kernel=3)
        smooth_val_is_brain_img_list = self.mean_filter(self.is_brain_val_img_list, weight=0.6, kernel=3)
        smooth_val_no_brain_img_list = self.mean_filter(self.no_brain_val_img_list, weight=0.6, kernel=3)
        smooth_g_loss_list = smooth_g_loss_list[:len(self.iterations_list)]
        smooth_d_loss_list = smooth_d_loss_list[:len(self.iterations_list)]
        smooth_g_img_list = smooth_g_img_list[:len(self.iterations_list)]
        smooth_val_img_list = smooth_val_img_list[:len(self.epoch_list)]
        smooth_val_is_brain_img_list = smooth_val_is_brain_img_list[:len(self.epoch_list)]
        smooth_val_no_brain_img_list = smooth_val_no_brain_img_list[:len(self.epoch_list)]

        plt.plot(self.iterations_list, smooth_g_loss_list, 'g-', label='g_loss')
        plt.plot(self.iterations_list, smooth_d_loss_list, 'r-', label='d_loss')
        plt.legend()
        plt.savefig(os.path.join(self.opt.save_path, 'loss_full_gan.png'), dpi=1000)
        plt.clf()

        plt.plot(self.iterations_list, smooth_g_img_list, 'r-', label='g_img')
        plt.legend()
        plt.savefig(os.path.join(self.opt.save_path, 'loss_full_recon.png'), dpi=1000)
        plt.clf()

        plt.plot(self.epoch_list, smooth_val_img_list, 'r-', label='val_img')
        plt.plot(self.epoch_list, smooth_val_is_brain_img_list, 'g-', label='val_img_is_brain')
        plt.plot(self.epoch_list, smooth_val_no_brain_img_list, 'b-', label='val_img_no_brain')
        plt.legend()
        plt.savefig(os.path.join(self.opt.save_path, 'loss_val_recon.png'), dpi=1000)
        plt.clf()

        np.save(self.g_loss_list_path, np.array(self.g_loss_list))
        np.save(self.d_loss_list_path, np.array(self.d_loss_list))
        np.save(self.g_img_list_path, np.array(self.g_img_list))
        np.save(self.val_img_list_path, np.array(self.val_img_list))
        np.save(self.is_brain_val_img_list_path, np.array(self.is_brain_val_img_list))
        np.save(self.no_brain_val_img_list_path, np.array(self.no_brain_val_img_list))

    def eval_and_save(self, val_input, val_target, val_is_brain, epoch):
        """
        Args:
            val_input: b, 1, h, w, d
            val_target: b, 1, h, w, d
            epoch:
        Returns:
        """
        with torch.no_grad():
            torch.cuda.synchronize()
            self.netG.eval()

            loss = 0.0
            loss_is_brain = 0.0
            count_is_brain = 0
            loss_no_brain = 0.0
            count_no_brain = 0
            val_out = torch.FloatTensor(val_input.size(0) * 2, 1, val_input.size(2), val_input.size(3)).fill_(0)
            for idx in range(val_input.shape[0]):
                single_val_input = val_input[idx, ...].unsqueeze(0).to(self.device)
                single_val_target = val_target[idx, ...].unsqueeze(0).to(self.device)
                is_brain = val_is_brain[idx].data.item()

                if not self.opt.use_amp:
                    if self.opt.use_ddp:
                        fake = self.netG.module.forward(single_val_input)[-1]
                    else:
                        fake = self.netG(single_val_input)[-1]
                else:
                    with autocast():
                        if self.opt.use_ddp:
                            fake = self.netG.module.forward(single_val_input)[-1]
                        else:
                            fake = self.netG(single_val_input)[-1]

                # pixel_loss = self.pixel_loss.pixel_loss(fake, single_val_target, loss_type=self.opt.loss_type)
                pixel_loss = self.pixel_loss.eval_pixel_loss(fake, single_val_target, loss_type=self.opt.loss_type)
                loss += pixel_loss.data.item()
                if is_brain:
                    loss_is_brain += pixel_loss.data.item()
                    count_is_brain += 1
                else:
                    loss_no_brain += pixel_loss.data.item()
                    count_no_brain += 1

                visual_index = fake.shape[-1] // 2
                val_out[idx * 2 + 0, :, :, :].copy_(single_val_target.data[0, :, :, :, visual_index])
                val_out[idx * 2 + 1, :, :, :].copy_(fake.data[0, :, :, :, visual_index])
            val_out += self.opt.mean

            self.epoch_list.append(epoch)
            self.val_img_list.append(loss / val_input.shape[0])
            self.is_brain_val_img_list.append(loss_is_brain / (count_is_brain + 1e-5))
            self.no_brain_val_img_list.append(loss_no_brain / (count_no_brain + 1e-5))

            vutils.save_image(val_out, os.path.join(self.opt.save_path, 'full_epoch{:05}.png'.format(epoch)), nrow=8,
                              normalize=True, range=(0, 1))

            self.netG.train()

            if isinstance(self.netG, torch.nn.DataParallel):
                torch.save({
                    'gen': self.netG.module.state_dict(),
                    'dis': self.netD.module.state_dict() if self.opt.use_gan_loss else None,
                    'gen_optim': self.optimizerG.state_dict(),
                    'dis_optim': self.optimizerD.state_dict() if self.opt.use_gan_loss else None
                }, os.path.join(self.opt.save_path, 'full_epoch{:05}.pth'.format(epoch)))
            else:
                torch.save({
                    'gen': self.netG.state_dict(),
                    'dis': self.netD.state_dict() if self.opt.use_gan_loss else None,
                    'gen_optim': self.optimizerG.state_dict(),
                    'dis_optim': self.optimizerD.state_dict() if self.opt.use_gan_loss else None
                }, os.path.join(self.opt.save_path, 'full_epoch{:05}.pth'.format(epoch)))
            torch.cuda.synchronize()

    def load_model(self):
        """
        Returns:
        """
        if self.opt.latest_train_epoch >= 0:
            model_path = os.path.join(self.opt.save_path, 'full_epoch{:05}.pth'.format(self.opt.latest_train_epoch))
            print('loading latest model: {}'.format(model_path))
            state_dict = torch.load(model_path, map_location=self.device)

            new_state_dict_G = {}
            for k, v in state_dict['gen'].items():
                k = k.replace("module.", '')
                new_state_dict_G[k] = v
            self.netG.load_state_dict(new_state_dict_G)
            
            if self.opt.use_gan_loss:
                new_state_dict_D = {}
                for k, v in state_dict['dis'].items():
                    k = k.replace("module.", '')
                    new_state_dict_D[k] = v
                self.netD.load_state_dict(new_state_dict_D)

            '''
            try:
                self.optimizerG.load_state_dict(state_dict['gen_optim'])
                if self.opt.use_gan_loss:
                    self.optimizerD.load_state_dict(state_dict['dis_optim'])
            except Exception as e:
                print(e)
            '''
        else:
            print('no latest model need load!!!')

    def test(self, ct_path, save_path, cta_path=None):
        """
        test
        Args:
            ct_path:
            save_path:
            cta_path:
        Returns:
        """
        self.netG.eval()

        ori_ct_data = read_nii_to_np(ct_path)
        if cta_path is not None:
            ori_cta_data = read_nii_to_np(cta_path)

        ct_data = ori_ct_data.astype(np.int16)
        if cta_path is not None:
            ori_cta_data = ori_cta_data.astype(np.int16)

        ct_data = center_crop(ct_data, center_size=432)  # 432, 432, ori_c
        if cta_path is not None:
            ori_cta_data = center_crop(ori_cta_data, center_size=432)  # 432, 432, ori_c

        # [x, y, z], 0 - 1
        ct_data_cache = normalize_nii_data(ct_data.astype(np.float32))  # 432, 432, ori_c
        if cta_path is not None:
            # [x, y, z], 0 - 1
            ori_cta_data = normalize_nii_data(ori_cta_data.astype(np.float32))  # 432, 432, ori_c
            ori_cta_data = center_crop(ori_cta_data, center_size=self.opt.center_in_size)  # 384, 384, ori_c

        loss = 0.0
        loss_is_brain = 0.0
        loss_no_brain = 0.0
        # 多块融合
        #      24
        # 24  384  24
        #      24
        assert ct_data_cache.shape[0] == 432 == ct_data_cache.shape[1]
        ct_data_center = ct_data_cache[24:-24, 24:-24, :].copy()

        ct_data_left = ct_data_cache[24:-24, :-48, :].copy()
        ct_data_right = ct_data_cache[24:-24, 48:, :].copy()

        ct_data_up = ct_data_cache[:-48, 24:-24, :].copy()
        ct_data_down = ct_data_cache[48:, 24:-24, :].copy()

        ct_data_list = [
            ct_data_center,
            ct_data_left, ct_data_right,
            ct_data_up, ct_data_down
        ]
        cta_data_list = []
        for ct_data in ct_data_list:
            cta_data = []
            for d in range(0, ct_data.shape[2], self.opt.center_depth):
                if d + self.opt.center_depth > ct_data.shape[2]:
                    break
                # [x, y, z]
                ct_data_d = ct_data[:, :, d: d + self.opt.center_depth]
                # [1, x, y, z]
                ct_data_d = torch.from_numpy(ct_data_d[None, :])
                ct_data_d = ct_data_d - self.opt.mean
                # [1, 1, x, y, z]
                ct_data_d = ct_data_d.unsqueeze(0)
                ct_data_d = ct_data_d.to(self.device)
                # [1, 1, x, y, z]
                fake = self.netG(ct_data_d)[-1] + self.opt.mean
                # [x, y, z]
                cta_data.append(fake.detach().data.cpu().numpy()[0][0])
            # [x, y, z]
            cta_data = np.concatenate(cta_data, axis=2)
            cta_data_list.append(cta_data)

        # 融合
        cta_data_center = cta_data_list[0]
        cta_data_left, cta_data_right = cta_data_list[1], cta_data_list[2]
        cta_data_up, cta_data_down = cta_data_list[3], cta_data_list[4]
        if cta_path is not None:
            min_depth = min(ori_cta_data.shape[2], cta_data_center.shape[2])
            loss = self.pixel_loss.eval_pixel_loss_numpy(
                ori_cta_data[:, :, :min_depth], cta_data_center[:, :, :min_depth])
            loss_is_brain = self.pixel_loss.eval_pixel_loss_numpy(
                ori_cta_data[:, :, 230:min_depth], cta_data_center[:, :, 230:min_depth])
            loss_no_brain = self.pixel_loss.eval_pixel_loss_numpy(
                ori_cta_data[:, :, :230], cta_data_center[:, :, :230])

        min_depth = min(ct_data_cache.shape[2], cta_data_center.shape[2])
        res_cta_data = np.zeros_like(ct_data_cache, dtype=np.float32)
        res_cta_data[24:-24, 24:-24, :min_depth] = cta_data_center[:, :, :min_depth]  # center
        res_cta_data[24:-24, :24, :min_depth] = cta_data_left[:, :24, :min_depth]  # left
        res_cta_data[24:-24, -24:, :min_depth] = cta_data_right[:, -24:, :min_depth]  # right
        res_cta_data[:24, 24:-24, :min_depth] = cta_data_up[:24, :, :min_depth]  # up
        res_cta_data[-24:, 24:-24, :min_depth] = cta_data_down[-24:, :, :min_depth]  # down
        # 融合部分重叠区域
        res_cta_data[24:-24, 24:24+12, :min_depth] = cta_data_left[:, 24:24 + 12, :min_depth] * 0.5 + \
                                                     res_cta_data[24:-24, 24:24+12, :min_depth] * 0.5  # left
        res_cta_data[24:-24, -24-12:-24, :min_depth] = cta_data_right[:, -24-12:-24, :min_depth] * 0.5 + \
                                                       res_cta_data[24:-24, -24-12:-24, :min_depth] * 0.5  # right
        res_cta_data[24:24+12, 24:-24, :min_depth] = cta_data_up[24:24+12, :, :min_depth] * 0.5 + \
                                                     res_cta_data[24:24+12, 24:-24, :min_depth] * 0.5  # up
        res_cta_data[-24-12:-24, 24:-24, :min_depth] = cta_data_down[-24-12:-24, :, :min_depth] * 0.5 + \
                                                       res_cta_data[-24-12:-24, 24:-24, :min_depth] * 0.5  # down

        res_cta_data_ = np.zeros_like(ori_ct_data, dtype=np.float32)
        res_cta_data_ = center_crop_fill(res_cta_data_, res_cta_data)
        # 增强对比度
        res_cta_data_unnorm = unnormalize_nii_data(res_cta_data_)

        write_np_to_nii(res_cta_data_unnorm, ct_path, save_path)

        return loss, loss_is_brain, loss_no_brain

    def train(self, rank, train_dataset: CustomCT2CTADataLoader, val_dataset: CustomCT2CTADataLoader):
        """
        train
        Args:
            rank:
            train_dataset:
            val_dataset:
        Returns:
        """
        train_dataset_batches, val_dataset_batches = \
            len(train_dataset) // self.opt.batch_size, len(val_dataset) // self.opt.batch_size

        if self.opt.use_ddp:
            dist.barrier()

        val_input, val_target, val_is_brain = None, None, None
        if rank == 0:
            val_count = 0
            val_nums = 20 * val_dataset.dataset.center_random_num
            for data_val in tqdm(val_dataset):
                # if (val_count % val_dataset.dataset.center_random_num == 0) and \
                #         (val_count <= val_nums):
                if (val_count <= val_nums):
                    val_input_, val_target_, val_is_brain_ = data_val["A"], data_val["B"], data_val["is_brain"]
                    if (val_input is None) or (val_target is None) or (val_is_brain_ is None):
                        val_input, val_target, val_is_brain = val_input_, val_target_, val_is_brain_
                    else:
                        val_input, val_target, val_is_brain = \
                            torch.cat((val_input, val_input_), dim=0), \
                            torch.cat((val_target, val_target_), dim=0), \
                            torch.cat((val_is_brain, val_is_brain_), dim=0)
                val_count += 1
            print("val batch size: {}".format(val_input.shape[0]))
            print("val is brain size: {}".format(val_is_brain.shape))

        if self.opt.use_amp:
            scaler_D = GradScaler()
            scaler_G = GradScaler()

        all_time = 0
        for epoch in range(self.opt.latest_train_epoch + 1, self.opt.n_epochs):
            train_dataset.set_epoch(epoch)
            start_time = time.time()
            with tqdm(total=train_dataset_batches) as process_bar:
                for i, data in enumerate(train_dataset):
                    A, B = data["A"], data["B"]
                    A, B = A.to(self.device), B.to(self.device)
                    ####################################################################################################
                    ####################################################################################################
                    # optim netD
                    torch.cuda.synchronize()
                    if self.opt.use_gan_loss:
                        requires_grad(self.netD, True)

                    if self.opt.use_gan_loss:
                        if not self.opt.use_amp:
                            out_list = self.netG(A)

                            fake = out_list[-1]
                            fake_detach = fake.detach()

                            loss_D = self.gan_loss.dis_loss(B, fake_detach, self.netD)

                            self.optimizerD.zero_grad()
                            loss_D.backward()
                            self.optimizerD.step()
                        else:
                            with autocast():
                                out_list = self.netG(A)

                                fake = out_list[-1]
                                fake_detach = fake.detach()

                                loss_D = self.gan_loss.dis_loss(B, fake_detach, self.netD)

                            self.optimizerD.zero_grad()
                            scaler_D.scale(loss_D).backward()
                            scaler_D.step(self.optimizerD)
                            scaler_D.update()
                    else:
                        out_list = self.netG(A)
                        fake = out_list[-1]

                    if self.opt.use_gan_loss:
                        requires_grad(self.netD, False)
                    torch.cuda.synchronize()
                    if self.opt.use_ddp:
                        dist.barrier()
                    ####################################################################################################
                    ####################################################################################################
                    # optim netG
                    torch.cuda.synchronize()

                    if self.opt.use_gan_loss:
                        if not self.opt.use_amp:
                            g_loss = self.gan_loss.gen_loss(None, fake, self.netD)
                            if self.opt.loss_type in ["l1", "l2"]:
                                pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type=self.opt.loss_type)
                                eval_pixel_loss = self.pixel_loss.eval_pixel_loss(fake, B, loss_type=self.opt.loss_type)
                            else:
                                # 前xxepoch使用l1, 后面使用l2
                                loss_type, loss_epoch = self.opt.loss_type.split("_")
                                loss_epoch = int(loss_epoch)
                                assert loss_type in ["l1"]
                                if epoch <= loss_epoch:
                                    pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type="l1")
                                    eval_pixel_loss = self.pixel_loss.eval_pixel_loss(fake, B, loss_type="l1")
                                else:
                                    pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type="l2")
                                    eval_pixel_loss = self.pixel_loss.eval_pixel_loss(fake, B, loss_type="l2")

                            loss_G = g_loss + \
                                     pixel_loss * self.opt.lambda_pixel

                            self.optimizerG.zero_grad()
                            loss_G.backward()
                            self.optimizerG.step()
                        else:
                            with autocast():
                                g_loss = self.gan_loss.gen_loss(None, fake, self.netD)
                                pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type=self.opt.loss_type)

                                loss_G = g_loss + \
                                         pixel_loss * self.opt.lambda_pixel

                            self.optimizerG.zero_grad()
                            scaler_G.scale(loss_G).backward()
                            scaler_G.step(self.optimizerG)
                            scaler_G.update()
                    else:
                        if self.opt.loss_type in ["l1", "l2"]:
                            pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type=self.opt.loss_type)
                            eval_pixel_loss = self.pixel_loss.eval_pixel_loss(fake, B, loss_type=self.opt.loss_type)
                        else:
                            # 前xxepoch使用l1, 后面使用l2
                            loss_type, loss_epoch = self.opt.loss_type.split("_")
                            loss_epoch = int(loss_epoch)
                            assert loss_type in ["l1"]
                            if epoch <= loss_epoch:
                                pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type="l1")
                                eval_pixel_loss = self.pixel_loss.eval_pixel_loss(fake, B, loss_type="l1")
                            else:
                                pixel_loss = self.pixel_loss.pixel_loss(fake, B, loss_type="l2")
                                eval_pixel_loss = self.pixel_loss.eval_pixel_loss(fake, B, loss_type="l2")

                        loss_G = pixel_loss * self.opt.lambda_pixel

                        self.optimizerG.zero_grad()
                        loss_G.backward()
                        self.optimizerG.step()

                    torch.cuda.synchronize()
                    if self.opt.use_ddp:
                        dist.barrier()
                    ####################################################################################################
                    ####################################################################################################
                    if self.opt.display_batch > train_dataset_batches:
                        self.opt.display_batch = train_dataset_batches * 9 // 10
                    if (self.ganIterations + 1) % self.opt.display_batch == 0 and rank == 0:
                        if self.opt.use_gan_loss:
                            self.vis_train_info(epoch, i, train_dataset_batches,
                                                {'d_loss': loss_D.data.item(),
                                                 'g_img': eval_pixel_loss.data.item(),
                                                 'g_vgg': 0.0,
                                                 'g_loss': g_loss.data.item()})
                            self.eval_and_save(val_input, val_target, val_is_brain, epoch)
                            self.write_train_info(
                                loss_D.data.item(), g_loss.data.item(),
                                eval_pixel_loss.data.item(), 0.0
                            )
                        else:
                            self.vis_train_info(epoch, i, train_dataset_batches,
                                                {'d_loss': 0.0,
                                                 'g_img': eval_pixel_loss.data.item(),
                                                 'g_vgg': 0.0,
                                                 'g_loss': 0.0})
                            self.eval_and_save(val_input, val_target, val_is_brain, epoch)
                            self.write_train_info(
                                0.0, 0.0,
                                eval_pixel_loss.data.item(), 0.0
                            )
                    # if self.opt.use_ddp:
                    #     dist.barrier()

                    process_bar.update(1)
                    self.ganIterations += 1
            end_time = time.time()
            spend_time = (end_time - start_time) / (60 * 60)
            all_time += spend_time
            if rank == 0:
                print('epoch: {}, spend time: {}'.format(epoch, spend_time))
            if self.opt.use_ddp:
                dist.barrier()
        if rank == 0:
            print("all time: {}".format(all_time))
