import os
import random

import SimpleITK as sitk
import cc3d
import cv2
import h5py
import imageio
import lmdb
import numpy as np
import pickle
from myplt import *
import torch
from monai.visualize import matshow3d
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import transforms
from scipy import ndimage
from modelGenesis.pytorch.utils import flip_a_lot, model_genesis_aug
from modelGenesis.pytorch.utils_2d import generate_pair_multi_channel
from seg_code_2d.seg_code_2d.util.img_mask_aug import *
from sklearn import preprocessing
import PIL.Image as pilimage
import matplotlib.pyplot as plt
# import pymrt
# import mygeometry

def circle_dilate(mask):
    # TODO 两次膨胀是否有必要？
    circle1 = np.zeros_like(mask)
    circle2 = np.zeros_like(mask)
    cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cont)):
        (x, y), radius = cv2.minEnclosingCircle(cont[i])
        center = (int(x), int(y))
        radius = int(radius)
        if radius < 10:
            radius = 10
        circle1 = cv2.circle(circle1, center, radius + 10, (255, 255, 255), -1)
        circle2 = cv2.circle(circle2, center, radius + 20, (255, 255, 255), -1)
    # circle1[mask==1]=0
    circle1 = minmax_norm(circle1)
    # circle2[mask==1] = 0
    circle2 = minmax_norm(circle2)
    # plt.imshow(image[1])
    # plt.show()
    # plt.imshow(mask)
    # plt.show()
    return circle1, circle2


def torchvision_crop(image, size=288, scale=(0.02, 1)):
    if random.random() < 0.5:
        augmentation = [
            transforms.RandomResizedCrop(size, scale=scale),
        ]
    else:
        augmentation = [
            transforms.RandomResizedCrop(size, scale=(0.02, 0.4)),
        ]
    tsf = transforms.Compose(augmentation)
    # image = torch.tensor(image[np.newaxis,...])
    image = torch.tensor(image, dtype=torch.float32)
    image = tsf(image)
    # image = image.squeeze()
    image = np.array(image)
    return image


def get_ct_stack(img_3d, num_layer):
    emerge_stack = np.zeros((img_3d.shape[1], img_3d.shape[2]))
    disappear_stack = np.zeros((img_3d.shape[1], img_3d.shape[2]))
    start = (len(img_3d) + 1) // 2 - (num_layer + 1) // 2
    end = (len(img_3d) + 1) // 2 + (num_layer - 1) // 2
    for i in range(start, end - 1):
        diff = img_3d[i] - img_3d[i + 1]
        # diff_blur = cv2.GaussianBlur(diff, (3, 3), 1)
        new = np.clip(diff, -180, 180)
        emerge = new.copy()
        emerge[emerge < 0] = 0
        disappear = new.copy()
        disappear[disappear > 0] = 0
        disappear = -disappear
        emerge_stack += emerge
        disappear_stack += disappear
    emerge_stack = np.clip(emerge_stack, 20, 1000)
    disappear_stack = np.clip(disappear_stack, 20, 1000)
    # plt.imshow(emerge_stack)
    # plt.show()
    # plt.imshow(img_3d[i + 1])
    # plt.show()
    return emerge_stack, disappear_stack


def get_pet_stack(img_3d, num_layer):
    pet_stacked = np.zeros((img_3d.shape[1], img_3d.shape[2]))
    img_3d_copy = img_3d.copy()
    start = (len(img_3d) + 1) // 2 - (num_layer + 1) // 2
    end = (len(img_3d) + 1) // 2 + (num_layer - 1) // 2
    threshold = 40
    for i in range(start, end):
        img = (255 * minmax_norm(img_3d_copy[i])).astype(np.uint8)
        ret, thresh11 = cv2.threshold(img, threshold, 255, type=cv2.THRESH_TOZERO)
        thresh11[thresh11 == 0] = threshold
        thres = cv2.adaptiveThreshold(thresh11, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 111, -17)
        pet_stacked += thres
    return pet_stacked
    # plt.imshow(thresh11)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(thres)
    # plt.show()


def z_norm(img):
    if img.max()>0.14:
        img_mean = np.mean(img[img>0.14])
        img_std = np.std(img[img>0.14])
    else:
        return img
    if img_std < 0.01:
        return (img - img_mean)/0.01
    img = (img - img_mean) / img_std
    return img
def ct_norm(img):
    img = np.clip(img,-1000,1000)+1000
    img = img / 500.0
    return img
def pet_norm(img):
    img = np.log2(img+1)
    img = z_norm(img)/2+1
    return img
def ct_norm_torch(img):
    img = torch.clip(img,-1000,1000)+1000
    img = img / 500.0
    return img
def pet_norm_torch(img):
    img = torch.log2(img+1)
    img = z_norm_torch(img)/2+1
    return img
def z_norm_torch(img):
    if img.max()>0.14:
        img_mean = torch.mean(img[img>0.14])
        img_std = torch.std(img[img>0.14])
    else:
        return img
    if img_std < 0.01:
        return (img - img_mean)/0.01
    img = (img - img_mean) / img_std
    return img

def minmax_norm(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min == 0:
        return img - img_min
    img = (img - img_min) / (img_max - img_min)
    return img

def concat(pet_all, pet):
    pet = pet[np.newaxis, ...]
    if pet_all is None:
        pet_all = pet
    else:
        pet_all = np.concatenate((pet_all, pet), axis=0)
    return pet_all

def crop_alot(imglist, center, size):
    for i in range(len(imglist)):
        imglist[i] = crop(imglist[i], center, size)
    return imglist

def crop(img, center, size):
    x, y = center
    s = size // 2

    if x - s < 0:
        x = s
    if x + s > img.shape[1]:
        x = img.shape[1] - s
    if y - s < 0:
        y = s
    if y + s > img.shape[1]:
        y = img.shape[1] - s
    x = int(x)
    y = int(y)

    if len(img.shape) > 2:
        return img[:, x - s:x + s, y - s:y + s]
    else:
        return img[x - s:x + s, y - s:y + s]

def crop_pos(shape, center, size):
    x, y = center
    s = size // 2

    if x - s < 0:
        x = s
    if x + s > shape[1]:
        x = shape[1] - s
    if y - s < 0:
        y = s
    if y + s > shape[1]:
        y = shape[1] - s
    x = int(x)
    y = int(y)
    return x - s, x + s, y - s, y + s

def save_pic(path, image, gt, pred,fnum_range=(0,30)):
    final_sample_image = np.zeros((pred.shape[2], pred.shape[2]))
    pred_save = image.data.cpu().clone().detach()
    pred_save = np.array(pred_save[0, ...])
    pred_save = np.rot90(pred_save,1,(1,2))
    for i_channel in range(pred_save.shape[0]):
        temp = pred_save[i_channel]
        temp = 255 * minmax_norm(temp)
        final_sample_image = np.concatenate((final_sample_image, temp), axis=0)
    ch_diff = pred.shape[1] - image.shape[1]
    if ch_diff >0:
        for _ in range(ch_diff):
            final_sample_image = np.concatenate((final_sample_image,
                                                 np.zeros_like(temp)), axis=0)


    final_sample_pred = np.zeros((pred.shape[2], pred.shape[2]))
    pred_save = pred.data.cpu().clone().detach()
    pred_save = np.array(pred_save[0, ...])
    pred_save = np.rot90(pred_save, 1,(1, 2))
    for i_channel in range(pred_save.shape[0]):
        temp = pred_save[i_channel]
        temp = 255 * minmax_norm(temp)
        final_sample_pred = np.concatenate((final_sample_pred, temp), axis=0)
    if ch_diff <0:
        for _ in range(-ch_diff):
            final_sample_pred = np.concatenate((final_sample_pred,
                                                 np.zeros_like(temp)), axis=0)


    final_sample_gt = np.zeros((pred.shape[2], pred.shape[2]))
    pred_save = gt.data.cpu().clone().detach()
    pred_save = np.array(pred_save[0, ...])
    pred_save = np.rot90(pred_save, 1,(1, 2))
    for i_channel in range(pred_save.shape[0]):
        temp = pred_save[i_channel]
        temp = 255 * minmax_norm(temp)
        final_sample_gt = np.concatenate((final_sample_gt, temp), axis=0)
    if ch_diff <0:
        for _ in range(-ch_diff):
            final_sample_gt = np.concatenate((final_sample_gt,
                                                 np.zeros_like(temp)), axis=0)

    final_sample = np.concatenate((final_sample_image, final_sample_gt, final_sample_pred), axis=1)
    final_sample = final_sample.astype(np.uint8)
    # plt.imshow((final_sample))
    # plt.show()
    l,m = fnum_range
    file_name = str(random.randint(l, m)) + '.png'
    imageio.imwrite(os.path.join(path, file_name),
                    final_sample)
    return

def calc_patient_pet(dpath, name="train"):
    lmdb_path = os.path.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    length = pickle.loads(txn.get(b'__len__'))
    keys = pickle.loads(txn.get(b'__keys__'))


def gen_fake_mask_3d(pet,mask,lo_rate):
    #TODO 卡不同阈值，或者adaptive阈值
    # 低亮度和高亮度各弄一次
    # 3d连通域一起删掉，还是随机删除2D
    # pet = pet.transpose(2,0,1)
    mask_hi = np.zeros_like(pet)
    is_lo = random.random()
    if is_lo<lo_rate:
        mask_lo=np.zeros_like(pet)
    mask_hi_dilate = np.zeros_like(pet)

    for i in range(pet.shape[0]):
        threshold = 40
        img = (255 * minmax_norm(pet[i])).astype(np.uint8)
        ret, thresh11 = cv2.threshold(img, threshold, 255, type=cv2.THRESH_TOZERO)
        thresh11[thresh11 == 0] = threshold
        # ps(thresh11)
        thres = cv2.adaptiveThreshold(thresh11, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 111, -17)
        # ps(thres)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
        mask_hi_dilate = cv2.dilate(thres,kernel,5)
        mask_hi_dilate = minmax_norm(mask_hi_dilate)
        # ps(mask_hi_dilate)
        mask_hi[i] = thres

    if is_lo < lo_rate:
        for i in range(pet.shape[0]):
            img = (255 * minmax_norm(pet[i])).astype(np.uint8)
            threshold = 60
            ret, thresh11 = cv2.threshold(img, threshold, 255, type=cv2.THRESH_TOZERO_INV)
            # ps(thresh11)
            threshold2 = 15
            ret, thresh22 = cv2.threshold(thresh11, threshold2, 255, type=cv2.THRESH_TOZERO)
            thresh22[thresh22 == 0] = threshold2
            # ps(thresh22)
            thres = cv2.adaptiveThreshold(thresh22, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 11, -5)
            # ps(thres)
            thres = thres-mask_hi_dilate*thres
            mask_lo[i] = thres
        mask_lo = minmax_norm(mask_lo)
        if random.random() < 0.5:
            mask_lo = np.bitwise_or(mask_lo.astype(np.bool), mask.astype(np.bool)).astype(np.uint8)
        fake_mask = random_remove_3d(minmax_norm(mask_lo),0.2)
    else:
        mask_hi = minmax_norm(mask_hi)
        if random.random() < 0.5:
            mask_hi = np.bitwise_or(mask_hi.astype(np.bool), mask.astype(np.bool)).astype(np.uint8)
        fake_mask = random_remove_3d(minmax_norm(mask_hi), 0.5)



    fake_mask = random_di_or_ero(fake_mask)
    return fake_mask

def random_di_or_ero(mask):
    rand = random.random()
    semisizes = tuple(np.random.randint(1,3,3))

    if rand < 0.33:
        return mask
    elif rand <0.66:
        return dilate_3d(mask,(7,7,7),semisizes=semisizes,itr=random.randint(1,3)).astype(np.uint8)
    else:
        return erode_3d(mask, (7, 7, 7), semisizes=semisizes, itr=random.randint(1,2)).astype(np.uint8)
def check_componnent_3d_lzt(mask, pet=None, min_size=25, max_size=1e8, pet_min=1):
    """
    检查全身分割结果，去掉不符合的病灶
    """
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    # cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint16))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)

        if np.sum(label_temp[:]) <= min_size:  # 体积太小
            # mask[label_temp] = 0
            label[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
    #     if np.sum(label_temp[:]) >= max_size:  # 体积太大
    #         mask[label_temp] = 0
    #         # print(np.sum(label_temp[:]))
    #         continue
    #     # 获取区域在z轴上的范围，如果不连续则去掉
    #     label_temp_z = np.sum(np.sum(label_temp, axis=2), axis=1)
    #     label_temp_z = label_temp_z > 0
    #     if np.sum(label_temp_z) <= 1:  # z轴上不连续
    #         mask[label_temp] = 0
    #         continue
    #     if pet is not None:
    #         if np.max(pet[label_temp][:]) <= pet_min:  # 均值太小
    #             mask[label_temp] = 0
    #             continue
    mask = (label>0).astype(np.uint8)
    return mask
def random_remove_3d(mask,keep_ratio=0.5):
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    # cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        lbsum = np.sum(label_temp[:])
        if lbsum <= 27:  # 体积太小
            mask[label_temp] = 0
        if lbsum > 156800:
            mask[label_temp] = 0
    num_to_keep = random.randint(5,30)
    if num_to_keep>num:
        num_to_keep=num
    if num<=1:
        return mask
    rand = np.random.randint(1,num,num_to_keep)
    mask = mask*np.isin(label,rand).astype(np.uint8)
    return mask
def dilate_3d(img,shape,semisizes=None,itr=1,):
    assert shape[0]%2 == 1
    if semisizes is None:
        semisizes = (shape[0] - 1) // 2
    struct = create_sphere_struct(shape,semisizes)
    img = ndimage.binary_dilation(img, structure=struct,iterations=itr)
    return img
def erode_3d(img,shape,semisizes=None,itr=1):
    assert shape[0]%2 == 1
    if semisizes is None:
        semisizes = (shape[0] - 1) // 2
    struct = create_sphere_struct(shape,semisizes)
    img = ndimage.binary_erosion(img, structure=struct,iterations=itr)
    return img
def create_sphere_struct(shape, radius):
    position = ((shape[0]-1)//2,(shape[0]-1)//2,(shape[0]-1)//2)
    # print(position)
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    if isinstance(radius,tuple):
        semisizes  = radius
    else:
        semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    # the inner part of the sphere will have distance below 1
    return arr <= 1.0

def concat_alot_3d(img_list, final_img=None):
    l,h,w,c = img_list[0].shape[0],img_list[0].shape[1],img_list[0].shape[2],len(img_list)
    final_img = np.zeros((l,h,w,c))
    for i in range(len(img_list)):
        final_img[...,i] = img_list[i]
    return final_img

if __name__ == "__main__":
    import SimpleITK as sitk
    image = sitk.ReadImage(r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/CTres.nii.gz')
    image = sitk.GetArrayFromImage(image)
    # pet = image[150:160]/
    # a = mygeometry.ellipsoid(5,(2,1))
    ct =  image[150]
    ct = ct.clip(-60,100)
    laplacian = cv2.Laplacian(ct,cv2.CV_32F)
    ct = minmax_norm(ct)
    import time
    tt = time.time()
    image = sitk.ReadImage(r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/PET.nii.gz')
    print(tt-time.time())
    p = r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/'
    a = os.listdir(r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/')
    for i in a[10:]:
        tt = time.time()
        b = os.listdir(os.path.join(p,i))
        image = sitk.ReadImage(os.path.join(r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/',
                                            i,b[0],'PET.nii.gz'))
        image = sitk.GetArrayFromImage(image)
        print(tt - time.time())


def get_small_lesion(mask):
    cc,num = cc3d.connected_components(mask,connectivity=18,return_N=True)
    size_list = []
    for i in range(1,num+1):
        voxel_size = np.isin(cc,i).astype(np.uint8).sum()
        size_list.append((voxel_size,i))
    size_list.sort(key=lambda x:x[0],reverse=True)
    large_to_keep = [size_list[0][1]]
    largest_size = size_list[0][0]
    for i in range(1,len(size_list)):

        if size_list[i][0]>0.5*largest_size:
            large_to_keep.append(size_list[i][1])
    large = np.isin(cc,np.asarray(large_to_keep))
    small = np.bitwise_not(large).astype(np.uint8)
    return small

