import os
import random
import time

import SimpleITK
import cv2
import h5py
import lmdb
import numpy as np
import pickle
import SimpleITK as sitk
import pandas
import torch
from torch.utils import data
from torchvision import transforms as T
from monai.visualize import matshow3d
from modelGenesis.pytorch.utils import flip_a_lot, model_genesis_aug
from modelGenesis.pytorch.utils_2d import generate_pair_multi_channel, LocalPixelShuffling, \
    data_augmentation_2d_multi_channel
from seg_code_2d.seg_code_2d.util.img_mask_aug import *
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils_data_proc import *
from lmdb_image import LMDB_Image
from myplt import *
import cc3d
import compress_pickle

class ImageFolder_3D_3step(data.Dataset):
    def __init__(self, h5list, z_len, db_path=None, image_size=512, mode='train', augmentation_prob=0.4,
                 crop_center=None, config=None,fake_mask=None):
        """Initializes image paths and preprocessing module."""
        # GT : Ground Truth
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.path_and_id = pandas.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/path_and_id.csv')

        self.db_path = db_path
        self.env = None
        self.fake_mask = fake_mask
        self.h5_paths = h5list
        self.test_crop_center = crop_center
        self.mode = mode
        assert z_len % 2 == 0, 'z_len must be even'
        self.config = config
        self.image_size = image_size
        self.z_len = z_len
        self.augmentation_prob = augmentation_prob
        self.norm = 'z-score'  # 标准化的三种选择： False-用全身统计结果做z标准化； 'z-score'-z标准化； 'normal'-归一化；
        if not self.norm:
            print('标准化方法：3D全身标准化')
        else:
            print('标准化方法：2D的' + self.norm)
        # if mode=='pretrain':
        #     self.lps = LocalPixelShuffling(image_size, 10)

    def _concat(self, pet_all, pet):
        pet = pet[np.newaxis, ...]
        if pet_all is None:
            pet_all = pet
        else:
            pet_all = np.concatenate((pet_all, pet), axis=0)
        return pet_all



    def _dilate(self, img, kernel_size, itr):
        # TODO z轴扩大
        # kernel = np.ones((2, 2), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        mask_dilate = np.zeros(img.shape, dtype=np.uint8)
        for i in range(len(img)):
            mask_dilate[i] = cv2.dilate(img[i], kernel, iterations=itr)

        return mask_dilate

    def __getitem__(self, index):
        if self.db_path is not None:
            if self.env is None:
                self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                                     readonly=True, lock=False,
                                     readahead=False, meminit=False)
                with self.env.begin() as txn:
                    self.length = pickle.loads(txn.get(b'__len__'))
                    self.keys = pickle.loads(txn.get(b'__keys__'))

        """Reads an image from a file and preprocesses it and returns."""
        h5_path = self.h5_paths[index]
        filename = h5_path.split('/')[-1]
        path = h5_path[:len(h5_path) - len(filename)]
        patient_id = int(filename.split('.')[0])
        layer = int(filename.split('.')[1])

        # 按顺序读取h5文件里的图像，最后一个是GT
        image_all = None
        label_all = None
        pet_all = np.zeros((32,400,400),dtype=np.float32)
        ct_all = np.zeros((32,400,400),dtype=np.float32)
        mask_all = np.zeros((32,400,400),dtype=np.uint8)
        center_pos = None
        # tt = time.time()

        # nii_path = os.path.join(r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/',
        #                         self.path_and_id['path'][int(patient_id)])
        # pet_nii = sitk.ReadImage(os.path.join(nii_path,'PET.nii.gz'))
        # ct_nii = sitk.ReadImage(os.path.join(nii_path,'CTres.nii.gz'))
        # mask_nii = sitk.ReadImage(os.path.join(nii_path,'SEG.nii.gz'))
        # mask_nii = sitk.GetArrayFromImage(mask_nii)
        # pet_nii = sitk.GetArrayFromImage(pet_nii)
        # ct_nii = sitk.GetArrayFromImage(ct_nii)
        # max_layer = mask_nii.shape[0]
        # if layer + int(1 - (self.z_len) / 2) < 0:
        #     pet_all = pet_nii[:layer+int((self.z_len) / 2) + 1]
        #     ct_all = ct_nii[:layer+int((self.z_len) / 2) + 1]
        #     mask_all = mask_nii[:layer+int((self.z_len) / 2) + 1]
        #     pad = np.zeros((-int(1 - (self.z_len) / 2)-layer ,mask_nii.shape[1],mask_nii.shape[2]),
        #                    dtype=np.float32)
        #     pet_all = np.concatenate((pad,pet_all),axis=0)
        #     ct_all = np.concatenate((pad, ct_all), axis=0)
        #     mask_all = np.concatenate((pad, mask_all), axis=0)
        # elif layer + int((self.z_len) / 2) + 1 > max_layer:
        #     pet_all = pet_nii[layer+int(1 - (self.z_len) / 2):]
        #     ct_all = ct_nii[layer+int(1 - (self.z_len) / 2):]
        #     mask_all = mask_nii[layer+int(1 - (self.z_len) / 2):]
        #     pad = np.zeros(((layer+int((self.z_len) / 2) + 1-max_layer),mask_nii.shape[1],mask_nii.shape[2]),
        #                    dtype=np.float32)
        #     pet_all = np.concatenate((pet_all,pad),axis=0)
        #     ct_all = np.concatenate(( ct_all,pad), axis=0)
        #     mask_all = np.concatenate(( mask_all,pad), axis=0)
        # else:
        #     pet_all = pet_nii[layer+int(1 - (self.z_len) / 2):layer+int((self.z_len) / 2) + 1]
        #     ct_all = ct_nii[layer+int(1 - (self.z_len) / 2):layer+int((self.z_len) / 2) + 1]
        #     mask_all = mask_nii[layer+int(1 - (self.z_len) / 2):layer+int((self.z_len) / 2) + 1]
        #
        # mask = mask_nii[int(layer)]
        # mask_where = np.where(mask > 0)
        # if len(mask_where[0] > 0):
        #     rand = random.randint(0, len(mask_where[0]) - 1)
        #     x = mask_where[0][rand]
        #     y = mask_where[1][rand]
        #     s = self.image_size / 2
        #     x = x + random.randint(-s // 3, s // 3)
        #     y = y + random.randint(-s // 3, s // 3)
        #     center_pos = (x, y)
        # else:
        #     center_pos = (len(mask) / 2 + random.randint(-self.image_size // 3, self.image_size // 3),
        #                   len(mask) / 2 + random.randint(-self.image_size // 3, self.image_size // 3))
        #
        # mask_all = mask_all.transpose(1,2,0).astype(np.uint8)
        # pet_all = pet_all.transpose(1, 2, 0)
        # ct_all = ct_all.transpose(1, 2, 0)
        # tt = time.time()
        if self.db_path is not None:
            with self.env.begin() as txn:
                cnt = 0
                # t1 = time.time()
                keys = []
                tt = time.time()
                for iz in range(int(1 - (self.z_len) / 2), int((self.z_len) / 2) + 1):
                    name = (str(patient_id) + '.' + str(int(layer)+iz)).encode('ascii')
                    keys.append(name)
                cursor = txn.cursor()
                byteflow_dict = dict(cursor.getmulti(keys))


            for iz in range(int(1 - (self.z_len) / 2), int((self.z_len) / 2) + 1):
                name = (str(patient_id) + '.' + str(int(layer)+iz)).encode('ascii')
                if name in self.keys:
                    byteflow = byteflow_dict[name]
                    IMAGE = compress_pickle.loads(byteflow,compression='gzip')
                    img = IMAGE.get_image()
                    pet = img[..., 0].astype(np.float32)
                    ct = img[..., 1].astype(np.float32)
                    mask = img[..., 2].astype(np.uint8)

                else:
                    pet = np.zeros((400, 400))
                    ct = np.zeros((400, 400))
                    mask = np.zeros((400, 400))

                if iz == 0:
                    mask_where = np.where(mask > 0)
                    if len(mask_where[0] > 0):
                        rand = random.randint(0, len(mask_where[0]) - 1)
                        x = mask_where[0][rand]
                        y = mask_where[1][rand]
                        s = self.image_size / 2
                        x = x + random.randint(-s // 1.5, s // 1.5)
                        y = y + random.randint(-s // 1.5, s // 1.5)
                        center_pos = (x, y)
                    else:
                        center_pos = (len(pet) / 2 + random.randint(-self.image_size // 1.5, self.image_size // 1.5),
                                      len(pet) / 2 + random.randint(-self.image_size // 1.5, self.image_size // 1.5))
                pet_all[iz-int(1 - (self.z_len) / 2)] = pet
                ct_all[iz-int(1 - (self.z_len) / 2)] = ct
                mask_all[iz-int(1 - (self.z_len) / 2)] = mask
                cnt+=1

            # pet_all = np.array(pet_all).astype(np.float32)
            # ct_all = np.array(ct_all).astype(np.float32)
            # mask_all = np.array(mask_all).astype(np.uint8)

                # print('cpu',time.time()-tt)
                # print(time.time() - t1)
        else:
            for iz in range(int(1 - (self.z_len) / 2), int((self.z_len) / 2) + 1):
                h5_fname = path + patient_id + '.' + str(int(layer) + iz) + '.h5'
                if os.path.exists(h5_fname):
                    h5_data = h5py.File(h5_fname, 'r')
                    pet = h5_data['PET'][()]
                    ct = h5_data['CT'][()]
                    mask = h5_data['mask'][()].astype(np.uint8)
                    if pet.shape[0] != 400:
                        pet = cv2.resize(pet, (400, 400))
                    if ct.shape[0] != 400:
                        ct = cv2.resize(ct, (400, 400))
                    if mask.shape[0] != 400:
                        mask = cv2.resize(mask, (400, 400))
                else:
                    pet = np.zeros((400, 400))
                    ct = np.zeros((400, 400))
                    mask = np.zeros((400, 400))

                if iz == 0:
                    mask_where = np.where(mask > 0)
                    if len(mask_where[0] > 0):
                        rand = random.randint(0, len(mask_where[0]) - 1)
                        x = mask_where[0][rand]
                        y = mask_where[1][rand]
                        s = self.image_size / 2
                        x = x + random.randint(-s // 2, s // 2)
                        y = y + random.randint(-s // 2, s // 2)
                        center_pos = (x, y)
                    else:
                        center_pos = (len(pet) / 2 + random.randint(-self.image_size // 2, self.image_size // 2),
                                      len(pet) / 2 + random.randint(-self.image_size // 2, self.image_size // 2))

                pet_all = self._concat(pet_all, pet)
                ct_all = self._concat(ct_all, ct)
                mask_all = self._concat(mask_all, mask)

# mask_all = self._z_norm(mask_all)
        if (self.mode == 'train'):
            pet_all, ct_all, mask_all = crop_alot(
                [pet_all, ct_all, mask_all],
                center_pos, self.image_size)

        if self.mode == 'test':
            pet_all, ct_all, mask_all = crop_alot(
                [pet_all, ct_all, mask_all],
                self.test_crop_center, self.image_size)
        if self.mode == 'valid':
            pet_all, ct_all, mask_all = crop_alot(
                [pet_all, ct_all, mask_all],
                (200, 200), self.image_size)

        if self.fake_mask == True:
            # if random.random()<1:
            #
            # else:
            #     fake_mask = gen_fake_mask_3d(pet_all,mask_all,0.20)
            #     fake_mask_gt = np.bitwise_xor(fake_mask.astype(np.bool),mask_all.astype(np.bool))*np.bitwise_not(mask_all.astype(np.bool))
            #     fake_mask_gt = fake_mask_gt.astype(np.uint8)
            fake_mask = np.zeros_like(mask_all)
            fake_mask_gt = fake_mask

        #归一化
        pet_all = pet_norm(pet_all)
        ct_all = ct_norm(ct_all)

        # 分步分割
        aux_img_list = []
        aux_mask_list = []
        if self.config.dilate_mode == 'no':
            pass
        if self.config.dilate_mode == 'direct_whole':


            # plt.imshow(mask_dididilate)
            # plt.show()
            if mask_all.max()>0:
                mask_dilate = dilate_3d(mask_all, (5, 5, 5), semisizes=(1, 2, 2), itr=2).astype(np.uint8)
                mask_didilate = dilate_3d(mask_dilate, (7, 7, 7), semisizes=(1, 3, 3), itr=4).astype(np.uint8)
                mask_dididilate = dilate_3d(mask_didilate, (9, 9, 9), semisizes=(1, 4, 4), itr=3).astype(np.uint8)
                if self.fake_mask == True:
                    fake_mask_dilate = dilate_3d(fake_mask_gt, (7, 7, 7), semisizes=(1, 3, 3), itr=2).astype(np.uint8)
                mask_whole = np.ones_like(mask_all)
                aux_pair = [
                    (mask_dilate, ct_all, 0),
                    (mask_dilate, pet_all, 0),
                    (mask_didilate, np.clip(ct_all, 1.8, 2.2) * 12 - 21.6, 1),
                    (mask_dididilate, pet_all, 1),
                    (mask_whole, ct_all, 1),
                ]
                if self.fake_mask == True:
                    aux_pair.append((fake_mask_dilate, ct_all + pet_all, 0))

                if self.config.focus_loss == True:
                    for (a, _, _) in aux_pair:
                        aux_mask_list.append(a)
                for (a, b, empty_center) in aux_pair:
                    # 中间是否挖空
                    if empty_center == 1:
                        a_cp = a.copy()
                        a_cp[mask_all == 1] = 0
                        aux_img_list.append(a_cp * b)
                    else:
                        aux_img_list.append(a * b)
            else:
                # mask_whole = np.zeros_like(mask_all)
                zeros_array = np.zeros_like(mask_all)
                for iii in range(6):
                    aux_img_list.append(zeros_array)
                    aux_mask_list.append(zeros_array)



        # p_transform = random.random()  # 是否扩增
        # TODO 随机裁剪，z轴方向某些裁剪某些不裁剪。高斯模糊
        # 上下左右翻转，对比度变换

        if (self.mode == 'train'):
            pass
        #     # 扩增操作
            # aug flip
            aml,ail = len(aux_mask_list),len(aux_img_list)


            if self.fake_mask == True:
                list_of_imgs = \
                    flip_a_lot([pet_all, ct_all, mask_all,fake_mask,fake_mask_gt] + aux_mask_list + aux_img_list)
                pet_all, ct_all, mask_all,fake_mask,fake_mask_gt = list_of_imgs[:5]
                aux_mask_list, aux_img_list = list_of_imgs[5:5 + aml], list_of_imgs[5 + aml:]
            else:
                list_of_imgs = \
                    flip_a_lot([pet_all, ct_all, mask_all] + aux_mask_list + aux_img_list)
                pet_all, ct_all, mask_all = list_of_imgs[:3]
                aux_mask_list, aux_img_list = list_of_imgs[3:3 + aml], list_of_imgs[3 + aml:]
            # pet_all, ct_all = model_genesis_aug([pet_all, ct_all])
        #     # aug crop,高斯模糊
        #     if random.random()>0.33:
                # pet_all, ct_all = aug_a_lot_3d([pet_all.astype(np.float32), ct_all.astype(np.float32)])
        #
        # if self.mode == 'test':
        #     pass


        # 取出GT，转float32是因为ToTensor会自动把uint8的图像除255
        if self.fake_mask == True:
            image = np.concatenate((pet_all[...,np.newaxis],ct_all[...,np.newaxis],fake_mask[...,np.newaxis]),axis=3)
            # image = concat_alot_3d([pet_all,ct_all,fake_mask])
            GT = np.concatenate([mask_all[...,np.newaxis],fake_mask_gt[...,np.newaxis]] + [ail[...,np.newaxis] for ail in aux_img_list],axis=3)
            # GT = concat_alot_3d([mask_all]+[fake_mask_gt]+aux_img_list)
        else:
            image = concat_alot_3d([pet_all, ct_all])
            GT = concat_alot_3d([mask_all] + aux_img_list)



        GT = GT.transpose(3, 0, 1, 2)
        image = image.transpose(3, 0, 1, 2)

        # 检测是否有nan
        # if np.count_nonzero(np.isnan(image)) > 0:
        #     print('image has nan!!')


        image = torch.tensor(image, dtype=torch.float32)
        GT = torch.tensor(GT, dtype=torch.float32)
        # print(time.time()-tt)
        if self.mode == 'test':
            return layer, image, GT

        if self.config.focus_loss == True:
            if self.fake_mask == True:
                aux_mask = np.concatenate([mask_all[...,np.newaxis],fake_mask_gt[...,np.newaxis]] + [ail[...,np.newaxis] for ail in aux_mask_list], axis=3)
                # aux_mask = concat_alot_3d([mask_all]+[fake_mask_gt]+aux_img_list)
            else:
                aux_mask = concat_alot_3d([mask_all] + aux_mask_list, None)
            aux_mask = aux_mask.transpose(3, 0, 1, 2)
            aux_mask = torch.tensor(aux_mask, dtype=torch.uint8)

            # 获取大mask区域
            if mask_all.max()>0:
                mask_small = get_small_lesion(mask_all)
                mask_small = mask_small[np.newaxis,...]
                mask_small = torch.tensor(mask_small, dtype=torch.uint8)
            else:
                mask_small=torch.ones((1,GT.shape[1],GT.shape[2],GT.shape[3]),dtype=torch.uint8)

            # time.sleep(0.4)

            return image, GT, aux_mask, mask_small

        return  image, GT

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.h5_paths)




