#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
dir = []
dir.append(r'/data/newnas/ZSN/2022_miccai_petct/code/code_lzt')
dir.append(r'/data/newnas/ZSN/2022_miccai_petct/code/code_lzt/seg_code_3d/')
for item in dir:
    sys.path.append(item)
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from monai.networks.layers import Norm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import datetime
import os
import random
import time
from monai.networks.nets import Unet,RegUNet
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss,DiceFocalLoss,TverskyLoss,FocalLoss,GeneralizedWassersteinDiceLoss,MaskedDiceLoss
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
import ttach as tta
from tensorboardX import SummaryWriter
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from seg_code_3d.data_loader_3d import *
from prefetch_loader import DataPrefetcher
import segmentation_models_pytorch as smp

from seg_code_2d.seg_code_2d.loss.loss_weight import *
from seg_code_2d.seg_code_2d.util.TNSUCI_util import *
from seg_code_2d.seg_code_2d.util.evaluation import *
from seg_code_2d.seg_code_2d.util.scheduler import *
from seg_code_2d.seg_code_2d.util.misc import printProgressBar
import torch.utils.data.distributed
import torch.multiprocessing as mp


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=r'/data/newnas/ZSN/2022_miccai_petct/seg_result/whole_3D_Unet_multip_160_autolw_fold5-2/models/1epoch265.pkl', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default=None, type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# r'/data/newnas/ZSN/2022_miccai_petct/seg_result/whole_3D_Swin_multip_fold5-2/models/1epoch7.pkl'
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default="tcp://localhost:5678", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',default=True,type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=128, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

parser.add_argument('--Task_name', type=str, default='whole_3D_Unet_multip_160_autolw', help='DIR name,Task name')

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=160)
parser.add_argument('--arch', type=str, default='SwinUNETR')  # 模型框架
parser.add_argument('--encoder_name', type=str, default='timm-regnety_160')  # 编码结构

# training hyper-parameters
parser.add_argument('--img_ch', type=int, default=3)
parser.add_argument('--out_ch', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=300)  # 总epoch

parser.add_argument('--num_epochs_decay', type=int, default=10)  # decay开始的最小epoch数
parser.add_argument('--decay_ratio', type=float, default=0.01)  # 0~1,每次decay到1*ratio
parser.add_argument('--decay_step', type=int, default=80)  # epoch

parser.add_argument('--batch_size', type=int, default=8)  # 训多少个图才回调参数
parser.add_argument('--batch_size_test', type=int, default=12)  # 测试时多少个,可以设很大,但结果图就会很小
parser.add_argument('--num_workers', type=int, default=3)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--multiprocessing_distributed', type=bool, default=False)
# 设置学习率(chrome有问题,额外记录在一个txt里面)
# 注意,学习率记录部分代码也要更改
parser.add_argument('--lr', type=float, default=1.7e-4)  # 初始or最大学习率(单用lovz且多gpu的时候,lr貌似要大一些才可收敛)
parser.add_argument('--lr_low', type=float, default=1e-7)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)

parser.add_argument('--lr_cos_epoch', type=int, default=300)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
parser.add_argument('--lr_warm_epoch', type=int, default=0)  # warm_up的epoch数,一般就是10~20,为0或False则不使用

# optimizer param
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--augmentation_prob', type=float, default=0.5)  # 数据扩增的概率

parser.add_argument('--save_model_step', type=int, default=1)  # 多少epoch保存一次模型
parser.add_argument('--val_step', type=int, default=3)  #

parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=2)  # 跑第几折的数据

# ==== 加入全身data到训练集中 ====
# 抽取全身的数据，间隔插入训练集内
parser.add_argument('--use_whole_body', type=bool, default=True)
parser.add_argument('--whole_body_scale', type=int, default=5)  # 全身图像的比例
parser.add_argument('--whole_body_epoch', type=int, default=10)  # 每次插入全身数据训练几个epoch
# data-parameters


# result&save
parser.add_argument('--save_detail_result', type=bool, default=True)
parser.add_argument('--save_image', type=bool, default=True)  # 训练过程中观察图像和结果

parser.add_argument('--train_healthy_img_csv', type=str,
                    default=r'/data/newnas/ZSN/2022_miccai_petct/data/train_patient_healthy.csv')
# data-parameters
# 存放训练数据h5的文件夹
parser.add_argument('--train_lesion_img_csv', type=str,
                    default=r'/data/newnas/ZSN/2022_miccai_petct/data/train_sparce_lesion_h5.csv')  # 用于分折的csv表格
parser.add_argument('--val_img_csv', type=str,
                    default=r'/data/newnas/ZSN/2022_miccai_petct/data/val_img.csv')  # 用于分折的csv表格

parser.add_argument('--filepath_img', type=str,
                    default=r'/data/newnas/ZSN/2022_miccai_petct/data/h5_data/v1')
# result&save
parser.add_argument('--result_path', type=str, default=r'/data/newnas/ZSN/2022_miccai_petct/seg_result')  # 结果保存地址
parser.add_argument('--pretrain_w_dir', type=str, default=None)  # 结果保存地址
# parser.add_argument('--db_path', type=str, default=r'/data/newnas/ZSN/2022_miccai_petct/data/lmdb/train2ch.lmdb')  # 结果保存地址
parser.add_argument('--db_path', type=str, default=r'/data/medai05/PCa/train2ch_compressed_gzip.lmdb')
parser.add_argument('--z_len', type=int, default=32)  # 3D图像z轴长度
parser.add_argument('--lesion_num', type=int, default=10)  # 1个epoch训练多少张图像
parser.add_argument('--healthy_num', type=int, default=20000)  # 1个epoch训练多少张图像
parser.add_argument('--valid_num', type=int, default=300)  # 验证多少张图像

parser.add_argument('--focus_loss', type=bool, default=True)  #
parser.add_argument('--dilate_mode', type=str, default='direct_whole')  #
parser.add_argument('--fake_mask', type=bool, default=True)  #
# more param
parser.add_argument('--mode', type=str, default='train', help='train/test')  # 训练还是测试
parser.add_argument('--cuda_idx', type=int, default=3)  # 用几号卡的显存
parser.add_argument('--amp', type=bool, default=False)
parser.add_argument('--DataParallel', type=bool, default=False)  # 数据并行,开了可以用多张卡的显存,不推荐使用
parser.add_argument('--train_flag', type=bool, default=False)  # 训练过程中是否测试训练集,不测试会节省很多时间
parser.add_argument('--seed', type=int, default=None)  # 随机数的种子点，一般不变
parser.add_argument('--accumulate_gd', type=int, default=1)  # 随机数的种子点，一般不变

def main():
    args = parser.parse_args()
    # step2: 设置各种文件保存路径 -------------------------------------------------------
    # 结果保存地址，后缀加上fold
    args.result_path = os.path.join(args.result_path,
                                      args.Task_name + '_fold' + str(args.fold_K) + '-' + str(args.fold_idx))
    args.model_path = os.path.join(args.result_path, 'models')
    args.log_dir = os.path.join(args.result_path, 'logs')       # 在终端用 tensorboard --logdir=地址 指令查看指标
    args.log_pic_dir = os.path.join(args.result_path, 'logger_pic')
    args.writer_4SaveAsPic = dict(lr=[], loss=[], loss_DICE=[], loss_lovz=[], loss_BCE=[], score_val=[])
    # Create directories if not exist
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        os.makedirs(args.model_path)
        os.makedirs(args.log_dir)
        os.makedirs(args.log_pic_dir)
        os.makedirs(os.path.join(args.result_path, 'images'))
    args.record_file = os.path.join(args.result_path, 'record.txt')
    f = open(args.record_file, 'a')
    f.close()

    # 保存设置到txt
    print(args)
    f = open(os.path.join(args.result_path, 'config.txt'), 'w')
    for key in args.__dict__:
        print('%s: %s' % (key, args.__getattribute__(key)), file=f)
    f.close()
    args.train_lesion_list = pd.read_csv(args.train_lesion_img_csv).values.tolist()
    args.train_healthy_list = pd.read_csv(args.train_healthy_img_csv).values.tolist()
    args.valid_list = pd.read_csv(args.val_img_csv).values.tolist()
    args.train_lesion_list_orig = [i[0] for i in args.train_lesion_list]
    args.train_lesion_list = [args.filepath_img + os.sep + i[0] for i in args.train_lesion_list]
    args.p_max_len = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/patient_max_len.csv').values.tolist()
    # all_lesion_epoch_list = []
    # for _ in range(100):
    #     lesion_epoch_list = []
    #     for i in range(len(args.train_lesion_list_orig)):
    #         h5 = args.train_lesion_list_orig[i]
    #         layer = int(h5.split('.')[1])
    #         pid = int(h5.split('.')[0])
    #         rand = random.randint(-15,15)
    #         while rand+layer>=args.p_max_len[pid][0] or rand+layer<0:
    #             rand = random.randint(-15, 15)
    #         name = str(pid) + '.' + str(layer+rand) + '.h5'
    #         lesion_epoch_list.append(name)
    #     all_lesion_epoch_list.append(lesion_epoch_list)
    #
    # with open('all_lesion_epoch_list', 'wb') as f:
    #         pickle.dump(all_lesion_epoch_list,f)

    with open('/data/newnas/ZSN/2022_miccai_petct/code/code_lzt/seg_code_3d/all_lesion_epoch_list', 'rb') as f:
        args.all_lesion_epoch_list = pickle.load(f)

    # p_max_len = args.p_max_len

    # p_lesion = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_patient_lesion.csv')
    # p_lesion = p_lesion['pid']
    # p_lesion = list(p_lesion)
    # train_pl_no = []
    # for p in p_lesion:
    #     pl_in_train = []
    #     healthy_part = []
    #     for i in range(p_max_len[p][0]):
    #         name = str(p)+'.'+str(i)+'.h5'
    #         if name in args.train_lesion_list_orig:
    #             pl_in_train.append(name)
    #         else:
    #             healthy_part.append(name)
    #     final = []
    #     for i in healthy_part:
    #         layer_i = int(i.split('.')[1])
    #         add = True
    #         for j in pl_in_train:
    #             layer_j = int(j.split('.')[1])
    #             if abs(layer_i-layer_j)<16:
    #                 add = False
    #         if add:
    #             final.append(i)
    #             print(i)
    #     train_pl_no = train_pl_no+final

    # csv = pd.DataFrame(data=train_pl_no)
    # csv.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/patient_healthy_part_h5.csv', header='php', index=None)
    # train_pl_no = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/patient_healthy_part_h5.csv')
    # train_pl_no = list(train_pl_no['0'])
    # args.train_lesion_healthy = train_pl_no
    # train_le_he_epoch = []
    # for _ in range(100):
    #     sampled = random.sample(train_pl_no,2000)
    #     true_sampled = []
    #     for i in range(len(sampled)):
    #         p_i = int(sampled[i].split('.')[0])
    #         layer_i = int(sampled[i].split('.')[1])
    #         add = True
    #         for j in range(len(true_sampled)):
    #             if i is not j:
    #                 p_j = int(true_sampled[j].split('.')[0])
    #                 layer_j = int(true_sampled[j].split('.')[1])
    #                 if p_i==p_j:
    #                     if abs(layer_i-layer_j) < 16:
    #                         add = False
    #         if add:
    #             true_sampled.append(sampled[i])
    #     train_le_he_epoch.append(true_sampled)
    # with open('train_le_he_epoch', 'wb') as f:
    #     pickle.dump(train_le_he_epoch,f)
    with open('/data/newnas/ZSN/2022_miccai_petct/code/code_lzt/seg_code_3d/train_le_he_epoch', 'rb') as f:
        args.all_epoch_lesion_healthy_list = pickle.load(f)

    #
    # args.train_healthy_list = [i[0] for i in args.train_healthy_list]
    # random.shuffle(args.train_healthy_list)
    # random.shuffle(args.train_lesion_list)
    # with open('sftrain_healthy_list', 'rb') as f:
    # #     args.train_healthy_list = pickle.load(f)
    # with open('sftrain_lesion_list', 'rb') as f:
    #     args.train_lesion_list = pickle.load(f)
    # args.train_healthy_list_epoch = []
    # for i in range(len(args.train_healthy_list) // 160):
    #     args.train_healthy_list_epoch.append(args.train_healthy_list[i*160:(i+1)*160])
    # args.train_healthy_list_epoch.append(args.train_healthy_list[(i + 1) * 160:])
    # args.train_healthy_list_epoch = args.train_healthy_list_epoch[:3]
    #
    # all_epoch_healthy_list = []
    # for start in range(0, 32):
    #     for ep in args.train_healthy_list_epoch:
    #         epoch_healthy_list = []
    #         for p in ep:
    #             layer = start
    #             while layer<p_max_len[p][0]:
    #                 name = str(p) + '.' + str(layer) + '.h5'
    #                 epoch_healthy_list.append(name)
    #                 layer=layer+32
    #         all_epoch_healthy_list.append(epoch_healthy_list)
    #
    #
    # with open('all_epoch_healthy_list', 'wb') as f:
    #     pickle.dump(all_epoch_healthy_list,f)


    with open('/data/newnas/ZSN/2022_miccai_petct/code/code_lzt/seg_code_3d/all_epoch_healthy_list', 'rb') as f:
        args.all_epoch_healthy_list = pickle.load(f)

    # args.valid_list = [args.filepath_img + os.sep + i[0] for i in args.valid_list]
    true_sampled = []

    for i in range(len(args.valid_list)):
        h5=args.valid_list[i][0]
        layer_i = int(h5.split('.')[1])
        p_i = int(h5.split('.')[0])
        add = True
        for j in range(len(true_sampled)):
            p_j = int(true_sampled[j].split('.')[0])
            layer_j = int(true_sampled[j].split('.')[1])
            if p_i==p_j:
                if abs(layer_i-layer_j) < 8:
                    add = False
        if add:
            true_sampled.append(h5)
    random.shuffle(true_sampled)
    true_sampled=true_sampled[0:400]
    args.valid_list = [args.filepath_img + os.sep + i for i in true_sampled]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def sample_train_data(args,epoch):

    l_id = epoch%len(args.all_lesion_epoch_list)
    lh_id =epoch%len(args.all_epoch_lesion_healthy_list)
    h_id =epoch%len(args.all_epoch_healthy_list)

    special = []
    special_patient = ['722','362','674','882','842','528','12',
                       '35','805','595','770','722','994','597',
                       '621','297','658','916','313','352','675','623','276','758']
    for sp in special_patient:
        for ii in range(args.p_max_len[int(sp)][0]//32):
            special.append(sp+'.'+str(random.randint(0,24)+ii*32)+'.h5')
    for ii in range(5):
        special.append('722.'+str(340+random.randint(-40,40))+'.h5')
    special = [args.filepath_img + os.sep + i for i in special]
    train_list = args.all_lesion_epoch_list[l_id] + \
                 args.all_epoch_lesion_healthy_list[lh_id]+\
                 args.all_epoch_healthy_list[h_id]+special
    # if args.gpu==0:
    random.shuffle(train_list)
    dataset = ImageFolder_3D_3step(h5list=train_list, db_path=args.db_path,
                                   z_len=args.z_len, image_size=args.image_size, mode='train',
                                   augmentation_prob=args.augmentation_prob,config=args,fake_mask=args.fake_mask)
    # if args.gpu==1:
    #     dataset = ImageFolder_3D_3step(h5list=train_list, db_path=r'/data/newnas/ZSN/2022_miccai_petct/data/lmdb/train2ch.lmdb',
    #                                    z_len=args.z_len, image_size=args.image_size, mode='train',
    #                                    augmentation_prob=args.augmentation_prob,config=args,fake_mask=args.fake_mask)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_sampler.set_epoch(epoch)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  shuffle=(train_sampler is None), num_workers=args.num_workers,
                                  drop_last=True,pin_memory=True,sampler=train_sampler,
                                  prefetch_factor=2)
    return train_loader,train_sampler

def sample_valid_data(args,epoch,n):
    # rd = np.random.RandomState(epoch)
    # ids = rd.randint(0,len(args.valid_list), n)
    valid_list = args.valid_list

    dataset = ImageFolder_3D_3step(h5list=valid_list, db_path=args.db_path,
                                   z_len=args.z_len, image_size=args.image_size, mode='valid',
                                   augmentation_prob=args.augmentation_prob,config=args,fake_mask=args.fake_mask)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    valid_sampler.set_epoch(epoch)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  drop_last=True,pin_memory=True,sampler=valid_sampler,
                                  prefetch_factor=2)
    return valid_loader,valid_sampler



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank - gpu
        # rank = dist.get_rank()
        # print(rank)
        dist.init_process_group(backend=args.dist_backend,init_method="tcp://localhost:5678",
                                world_size=args.world_size, rank=gpu)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    # model = SwinUNETR(
    #     img_size=(32, args.image_size, args.image_size),
    #     in_channels=args.img_ch,
    #     out_channels=args.out_ch,
    #     feature_size=48,
    #     use_checkpoint=False,
    # )
    model = Unet(
        dimensions=3,
        in_channels=args.img_ch,
        out_channels=args.out_ch,
        channels=(96, 96*2, 96*2, 96*2, 96*2),
        strides=(2, 2, 2, 2),
        num_res_units=4,
        norm=Norm.BATCH,
    )
#     model=Unet(
#     spatial_dims=3,
#     in_channels=args.img_ch,
#     out_channels=args.out_ch,
#     channels=(96, 192, 384,768,1536),
#     strides=(2, 2,2,2,2),
#     num_res_units=2
# )

    # infer learning rate before changing batch size
    init_lr = args.lr


    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    # criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    # loss

    dicece_loss = DiceFocalLoss(sigmoid=True,lambda_focal=2,lambda_dice=1,).cuda(args.gpu)
    dicece_loss_notp = DiceFocalLoss(sigmoid=True,lambda_focal=5,lambda_dice=1,).cuda(args.gpu)
    # tversky_loss = TverskyLoss(sigmoid=True,alpha=0.6,beta=0.4).cuda(args.gpu)
    maskeddc_loss = MaskedDiceLoss(sigmoid=True,).cuda(args.gpu)
    maskeddc_loss_notp = MaskedDiceLoss(sigmoid=True,).cuda(args.gpu)
    focal_loss = FocalLoss()
    # lovasz_loss = lovasz_hinge
    mse_loss = nn.MSELoss().cuda(args.gpu)
    if args.focus_loss == True:
        if args.fake_mask==True:
            lw = AutomaticWeightedLoss(device=args.gpu, num=args.out_ch * 2-2-2)
        else:
            lw = AutomaticWeightedLoss(device=args.gpu, num=args.out_ch * 2 - 1)
    else:
        lw = AutomaticWeightedLoss(device=args.gpu, num=args.out_ch)


    grad_scaler = GradScaler(enabled=args.amp)
    optimizer = optim.Adam(list(model.parameters())+list(lw.parameters()),
                            init_lr,weight_decay=0.00004)

    # optimizer = optim.SGD(list(model.parameters()),args.lr,momentum=0.9,weight_decay=0.00004)
    # optionally resume from a checkpoint
    loss_record = []
    cur_lr = init_lr
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            cur_lr = checkpoint['cur_lr']
            loss_record = checkpoint['loss_record']
            print(loss_record)
            model.load_state_dict(checkpoint['state_dict'])
            # my_weight_list = [0.6379960513486017, 0.9884239790561571, 6.166797311516819, 12.486530254124482, 8.835988926865394, 14.559777705063457, 4.888590223728525, 6.4004751331305487, 8.55498018561832, 6.058162712772121, 7.844461987498791, 7.625244900176145]
            # my_weight_list = [1/(i**0.5) for i in my_weight_list]
            print(len(checkpoint['autolw']),args.out_ch * 2 -2-2)
            lw = AutomaticWeightedLoss(device=args.gpu, num=args.out_ch * 2 -2-2,weights_list=checkpoint['autolw'])
            optimizer = optim.Adam(list(model.parameters())+list(lw.parameters()),
                                   init_lr, weight_decay=0.00004)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        del checkpoint

    cudnn.benchmark = True

    used_h = []
    used_l=[]
    # loss_record = []
    list_criterion = [dicece_loss, dicece_loss_notp, mse_loss, maskeddc_loss, maskeddc_loss_notp]
    # cur_lr=adjust_learning_rate_step(optimizer, cur_lr*2.5, 0, args)
    print('world_size:',args.world_size)
    for epoch in range(args.start_epoch,args.num_epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        print(epoch)
        epochtime = time.time()
        train_loader,train_sampler= sample_train_data(args,epoch)
        # train for one epoch

        losse = train(train_loader, model, list_criterion,lw, optimizer,grad_scaler, epoch, args)
        dist.all_reduce(losse, op=dist.ReduceOp.SUM)
        print(lw.get_weights())
        lossa = float(losse.cpu().numpy())
        del losse
        loss_record.append(lossa/args.world_size)
        print(loss_record)
        if len(loss_record)>14:
            mavg1,mavg2 = 0,0
            for aloss in loss_record[-15:-8]:
                mavg1+=aloss
            for aloss in loss_record[-8:]:
                mavg2+=aloss
            if mavg1/7-mavg2/8 < 5e-3:
                # cur_lr = adjust_learning_rate_step(optimizer,cur_lr,epoch,args)
                loss_record=[]

        if (epoch%args.save_model_step == 0) and gpu==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'autolw': lw.get_weights_save(),
                'cur_lr': cur_lr,
                'loss_record':loss_record
            }, is_best=False, filename=os.path.join(args.model_path, '1epoch%d.pkl' % (epoch + 1)))
        if (epoch) % args.val_step == 0:
            valid_loader,valid_sampler = sample_valid_data(args,epoch,10)
            valid(valid_loader, model, list_criterion, epoch, args,gpu)
            del valid_loader,valid_sampler
        print('epoch time:',( time.time()-epochtime)/60)
        del train_loader, train_sampler


def valid(valid_loader, model, list_criterion, epoch, args,rank):
    model.train(False)
    model.eval()
    acc = 0.  # Accuracy
    FP = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    DC = 0.  # Dice Coefficient
    IOU = 0.  # IOU
    length = 0
    GT_cnt = torch.tensor(0,dtype=torch.int32).cuda()
    length = torch.tensor(0, dtype=torch.int32).cuda()
    loss_all_ct = 0
    loss_all_pet = 0
    result_tensor = None
    # model pre for each image
    detail_result = []  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
    with torch.no_grad():
        for i, sample in enumerate(valid_loader):
            (images, GT, _,_) = sample
            tt=time.time()
            images = images.cuda(non_blocking=True)
            GT = GT.cuda(non_blocking=True)
            GT_mask = GT[:, 0:1, ...]

            SR = model(images)
            SR_mask = SR[:, 0:1, ...]
            SR_mask = F.sigmoid(SR_mask)

            if args.save_image and (i % 12 == 0):  # 20张图后保存一次png结果
                save_pic(os.path.join(args.result_path, 'images'), images[:, :, args.z_len // 2 - 1, ...]
                         , GT[:, :, args.z_len // 2 - 1, ...], SR[:, :, args.z_len // 2 - 1, ...], fnum_range=(30, 50))

            SR_mask = SR_mask.data.cpu().numpy()
            GT_mask = GT_mask.data.cpu().numpy()

            for ii in range(SR_mask.shape[0]):
                SR_tmp = SR_mask[ii, :].reshape(-1)
                GT_tmp = GT_mask[ii, :].reshape(-1)

                SR_tmp = torch.from_numpy(SR_tmp).cuda()
                GT_tmp = torch.from_numpy(GT_tmp).cuda()

                # acc, se, sp, pc, dc, _, _, iou = get_result_gpu(SR_tmp, GT_tmp) 	# 少楠写的
                result_tmp1 = get_result_gpu(SR_tmp, GT_tmp)

                result_tmp1 = torch.tensor(list(result_tmp1),dtype=torch.float32).cuda()
                if result_tensor is None:
                    result_tensor = result_tmp1
                else:
                    result_tensor+=result_tmp1

                # print(result_tmp)
                if GT_tmp.max() == 0:
                    pass
                else:
                    GT_cnt += 1
                length += 1
    dist.all_reduce(GT_cnt,op=dist.ReduceOp.SUM)
    dist.all_reduce(length, op=dist.ReduceOp.SUM)
    dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
    acc = result_tensor[0] / length
    FP = result_tensor[1] / length/ args.z_len
    FN = result_tensor[2] / length/ args.z_len
    PC = result_tensor[3] / length
    DC = result_tensor[6] / GT_cnt
    IOU = result_tensor[7] / length

    if rank == 0:
        myprint('[Validation] Acc: %.3f, SE: %.3f, SP: %.3f, PC: %.3f, Dice: %.3f, IOU: %.3f' % (
            acc, FP, FN, PC, DC, IOU),args)



def myprint(*argss):
    """Print & Record while training."""
    text,args = argss
    print(text)
    if args.gpu == 0:
        f = open(args.record_file, 'a')
        print(text, file=f)
        f.close()

def train(train_loader, model, list_criterion,lw, optimizer,grad_scaler, epoch, args):
    dicece_loss,dicece_loss_notp,mse_loss,maskeddc_loss,maskeddc_loss_notp= list_criterion

    myprint('-----------------------%s-----------------------------' % args.Task_name,args)
    unet_path = os.path.join(args.model_path, 'best_unet_score.pkl')
    # writer = SummaryWriter(log_dir=args.log_dir)

    # switch to train mode
    model.train()
    Iter = 0
    # valid_record = np.zeros((1, 8))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU]

    myprint('Training...',args)

    epoch_loss = 0
    length = 0
    loss_sum = 0
    loss_sum_cnt=0
    train_len = len(train_loader)

    prefetcher = DataPrefetcher(train_loader, optimizer)
    (images, GT, aux_masks,small_mask) = prefetcher.next()
    i = 0
    while images is not None:
        i += 1
        # for i, sample in enumerate(self.train_loader):
        # current_lr = self.optimizer.param_groups[0]['lr']
        # print(current_lr)
        tt = time.time()
        # t = torch.ones(images.shape[0]).to(self.device).long()
        # SR : Segmentation Result
        with autocast(enabled=args.amp):
            # (_, images, GT) = sample
            # images = images.to(self.device)
            # GT = GT.to(self.device)
            if args.gpu==0:
                pass
            if args.gpu == 1:
                pass
            SR = model(images)
            GT_mask, SR_mask = GT[:, 0:1, ...], SR[:, 0:1, ...]
            if args.fake_mask is True:
                GT_fake_mask,SR_fake_mask = GT[:, 1:2, ...], SR[:, 1:2, ...]
                loss_fk_dicece = dicece_loss(SR_fake_mask, GT_fake_mask)
            # if torch.isnan(SR).any():
            #     print('SR nan')
            # if torch.isnan(GT).any():
            #     print('GT nan')
            # if torch.isnan(images).any():
            #     print('images nan')
            # if torch.isnan(small_mask).any():
            #     print('small_mask nan')
            # if GT_mask.max()>0:
            dcl=dicece_loss(SR_mask,GT_mask)
            # else:
            #     dcl = dicece_loss_notp(SR_mask,GT_mask)
                # if small_mask.sum()==0:
                #     mdcl = 0
                # else:
            mdcl = maskeddc_loss(SR_mask,GT_mask,small_mask)
            # if torch.isnan(dcl):
            #     dcl = 1
            #     print('dcl nan')
            # if torch.isnan(mdcl):
            #     mdcl = 1
            #     print('mdcl nan')
            # if torch.isnan(loss_fk_dicece):
            #     loss_fk_dicece = 1
            #     print('fk nan')
            loss_dicece =  dcl+mdcl*0.05

            # 辅助loss
            loss_aux_list = []
            start_ch = 2 if args.fake_mask is True else 1

            for i_seg in range(start_ch, GT.shape[1]-1):
                GT_1, SR_1 = GT[:, i_seg:i_seg + 1, ...], SR[:, i_seg:i_seg + 1, ...]
                loss_aux_list.append(mse_loss(SR_1, GT_1))
                if args.focus_loss == True:
                    foc = aux_masks[:, i_seg:i_seg + 1, ...]
                    if foc.max() > 0:
                        loss_aux_list.append(mse_loss(SR_1[foc == 1], GT_1[foc == 1]))
                    else:
                        loss_aux_list.append(mse_loss(SR_1, GT_1))




            # TODO 1、计算整幅图+亮区 2、计算黑区+亮区 3、整图+黑区+亮区
            for li in range(len(loss_aux_list)):
                if torch.isnan(loss_aux_list[li]):
                    print('li nan')
            # 自动学习权重
            # loss = self.lw(loss_lovz, loss_bi_BCE,loss_softdice)
            if args.fake_mask is True:
                loss = lw([loss_dicece*3]+[loss_fk_dicece] + loss_aux_list)
            else:
                loss = lw([loss_dicece] + loss_aux_list)
            # if torch.isnan(loss):
                # loss = 5.3
                # print('loss nan')
            epoch_loss += float(loss)
            loss = loss / args.accumulate_gd

            grad_scaler.scale(loss).backward()
        # Backprop + optimize
        loss_sum_cnt+=1
        if i % args.accumulate_gd == 0:
            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad()

        # print('gpu time:',time.time()-tt)

        # if i % 20 ==0:
        #     plt.imshow(SR_probs.cpu().detach().numpy()[0,0,2,...])
        #     plt.show()
        #     plt.imshow(GT.cpu().detach().numpy()[0, 0, 2, ...])
        #     plt.show()
        #     plt.imshow(images.cpu().detach().numpy()[0, 0, 2, ...])
        #     plt.show()
        #     plt.imshow(images.cpu().detach().numpy()[0, 1, 2, ...])
        #     plt.show()
        length += 1
        Iter += 1
        # writer.add_scalars('Loss', {'loss': loss}, Iter)

        if args.save_image and (i % 50 == 0):  # 20张图后保存一次png结果
            save_pic(os.path.join(args.result_path, 'images'), images[:, :, args.z_len // 2 - 1, ...]
                     , GT[:, :, args.z_len // 2 - 1, ...], SR[:, :, args.z_len // 2 - 1, ...])

        (images, GT, aux_masks, small_mask) = prefetcher.next()
        ttt = time.time()
        # trainning bar
        if (args.gpu == 0) and i%args.print_freq==0:
            if (not isinstance(mdcl,int)) and (not isinstance(dcl,int)):
                print_content = 'total:' + str(loss.data.cpu().numpy()) + \
                                '  DiceCE:' + str(dcl.data.cpu().numpy())+ \
                                '  mdcl:' + str(mdcl.data.cpu().numpy()) + \
                                'time:'+str(ttt-tt)
            else:
                print_content = 'total:' + str(loss.data.cpu().numpy()) + \
                                '  DiceCE:' + str(dcl)+ \
                                '  mdcl:' + str(mdcl) + \
                                'time:'+str(ttt-tt)


            # '  pet:' + str(loss_aux1.data.cpu().numpy())+ \
            # '  petct:' + str(loss_aux3.data.cpu().numpy())

            # print_content = 'batch_loss:' + str(loss.data.cpu().numpy())
            printProgressBar(i + 1, train_len, content=print_content)
        for lal in loss_aux_list:
            del lal
        del loss, SR, dcl, mdcl,SR_1,SR_mask,loss_dicece,loss_fk_dicece,loss_aux_list
    return torch.tensor(epoch_loss).cuda()/loss_sum_cnt





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    # if cur_lr<1e-7:
    #     cur_lr=1e-7
    print(cur_lr)
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
def adjust_learning_rate_step(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr/2.5
    print('lr reduce',cur_lr)
    # if cur_lr<1e-7:
    #     cur_lr=1e-7
    print(cur_lr)
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    main()
