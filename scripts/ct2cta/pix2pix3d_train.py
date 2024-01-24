# !/usr/bin/env python3
# coding=utf-8
import os
import sys
import glob
import argparse
from argparse import Namespace

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.append("../..")
from medical_projects.utils.io_file import create_dir
from medical_projects.ct2cta.datasets.dataset import CustomCT2CTADataLoader
from medical_projects.ct2cta.scripts.pix2pix3d import Pix2Pix3d


def genvalconf(train_opt, **kwargs):
    """
    Args:
        train_opt:
        **kwargs:
    Returns:
    """
    conf = Namespace(**vars(train_opt))
    attr_dict = train_opt.__dict__
    for key, value in attr_dict.items():
        if 'val' in key and key.split('_')[0] in attr_dict:
            setattr(conf, key.split('_')[0], value)

    for key in kwargs:
        setattr(conf, key, kwargs[key])

    return conf


def get_opt(
        dataset_root,
        vgg16_model_path,
        save_path,
        mean=0.0,
        is_train=True,
        gpu_ids='0'
):
    """
    get option
    Args:
        dataset_root:
        vgg16_model_path:
        mean:
        is_train:
    Returns:
    """
    def str2bool(v):
        """
        Args:
            v:
        Returns:
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            import argparse
            raise argparse.ArgumentTypeError('Boolean value expected.')

    create_dir(save_path)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_root', type=str, default='{}'.format(dataset_root))
    parser.add_argument('--dataset_type', type=str, default='center')
    parser.add_argument('--is_train', type=str2bool, default=is_train)
    parser.add_argument('--mean', type=float, default=mean)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=300)

    parser.add_argument('--in_size', type=int, default=128)
    parser.add_argument('--center_in_size', type=int, default=384)
    parser.add_argument('--center_depth', type=int, default=32)
    parser.add_argument('--model_type', type=str, default="stylegan2")
    parser.add_argument('--vgg16_model_path', type=str, default='{}'.format(vgg16_model_path))
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--display_batch', type=int, default=100000000)
    parser.add_argument('--save_path', type=str, default='{}'.format(save_path))
    parser.add_argument('--lambda_pixel', type=float, default=100.0)
    parser.add_argument('--lambda_vgg', type=float, default=10.0)

    parser.add_argument('--loss_type', type=str, default="l1")
    parser.add_argument('--use_gan_loss', type=str2bool, default=True)

    parser.add_argument('--gpu_ids', type=str, default='{}'.format(gpu_ids),
                        help='-1    0    0,1,2,3')
    parser.add_argument('--use_ddp', type=str2bool, default=True)
    parser.add_argument('--ddp_port', type=str, default='12355')
    parser.add_argument('--world_size', type=int, default=3)
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--use_amp', type=str2bool, default=False)

    parser.add_argument('--latest_train_epoch', type=int, default=-1)

    parser.add_argument('--use_checkpoint', type=str2bool, default=False)
    parser.add_argument('--generator_scale', type=float, default=1.0)
    parser.add_argument('--discriminator_scale', type=float, default=1.0)
    parser.add_argument('--use_adamw', type=str2bool, default=True)

    opt = parser.parse_args()

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    opt.world_size = len(gpu_ids)
    opt.batch_size = opt.batch_size * opt.world_size

    if opt.world_size <= 1:
        opt.use_ddp = False

    # 检查最新的训练权重
    create_dir(opt.save_path)
    model_paths = sorted(glob.glob(os.path.join(opt.save_path, '*.pth')))
    if len(model_paths) >= 1:
        model_epochs = [
            int(os.path.splitext(os.path.basename(model_path))[0].split('full_epoch')[1]) for model_path in model_paths]
        opt.latest_train_epoch = max(model_epochs)

    return opt


def setup(rank, world_size, port):
    """
    Args:
        rank:
        world_size:
        port:
    Returns:
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    Returns:
    """
    dist.destroy_process_group()


def main(rank, world_size, train_opt):
    """
    main
    Args:
        rank:
        world_size:
        train_opt:
    Returns:
    """
    val_opt = genvalconf(train_opt, is_train=False)
    print(train_opt)
    print(val_opt)

    device = torch.device(rank)
    use_ddp = train_opt.use_ddp

    if use_ddp:
        setup(rank, world_size, train_opt.ddp_port)

    torch.cuda.set_device(device)

    train_dataset, val_dataset = \
        CustomCT2CTADataLoader(train_opt, rank=rank), CustomCT2CTADataLoader(val_opt, rank=rank)

    train_model = Pix2Pix3d(train_opt)
    train_model.device = device
    train_model.load_model()
    train_model.parallelize()

    train_model.train(rank=rank, train_dataset=train_dataset, val_dataset=val_dataset)


if __name__ == '__main__':
    train_opt = get_opt(
        dataset_root="/storage/huayun/dataset_split",
        vgg16_model_path='./software/vgg16_reducedfc.pth',
        save_path="experiment",
        is_train=True,
        gpu_ids='0,1,2'
    )

    world_size = train_opt.world_size

    if train_opt.use_ddp:
        print("use ddp!!!")
        mp.spawn(main, args=(world_size, train_opt), nprocs=world_size, join=True)
    else:
        main(0, world_size, train_opt)
