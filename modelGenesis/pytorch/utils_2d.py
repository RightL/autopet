from __future__ import print_function
import math
import os
import random
import copy
import time

import scipy
import imageio
import string
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def flip_a_lot_2d(x_list, prob=0.5):
    # augmentation by flipping
    cnt = 2
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1])
        for i in range(len(x_list)):
            x_list[i] = np.flip(x_list[i], axis=degree)
        cnt = cnt - 1
    return x_list

def data_augmentation_2d_multi_channel(x, y, prob=0.5):
    # augmentation by flipping
    #TODO 每个channel一起转还是可以分开转？
    cnt = 2
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1]) #TODO 待验证
        for i_channel in range(x.shape[0]):
            x[i_channel] = np.flip(x[i_channel], axis=degree)
            y[i_channel] = np.flip(y[i_channel], axis=degree)
        cnt = cnt - 1

    return x, y


def data_augmentation_2d(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1]) #TODO 待验证
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation_multi_channel(x, prob=0.5):
    nonlinear_x = copy.deepcopy(x)
    for i_channel in range(x.shape[0]):
        if random.random() >= prob:
            continue
        if nonlinear_x[i_channel].sum()==0:
            continue
        else:
            nonlinear_x[i_channel] = nonlinear_transformation(x[i_channel])

    return nonlinear_x

def nonlinear_transformation(x):
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    # xpoints = [p[0] for p in points]
    # ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=10000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


# TODO 移动像素，除了层数以外，pet,ct,ct_stack，organ可以有，PET_stack待验证，可以某些c有，有些c没有
def local_pixel_shuffling_2d_multi_channel(x, prob=0.5):
    local_shuffling_x = copy.deepcopy(x)
    for i_channel in range(x.shape[0]):
        if random.random() >= prob:
            continue
        local_shuffling_x[i_channel] = local_pixel_shuffling_2d(x[i_channel])

    return local_shuffling_x

def local_pixel_shuffling_2d(x):
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(2, 7)
        block_noise_size_y = random.randint(2, 7)

        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)

        window = orig_image[noise_x:noise_x+block_noise_size_x,
                               noise_y:noise_y+block_noise_size_y,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y))
        image_temp[noise_x:noise_x+block_noise_size_x,
                      noise_y:noise_y+block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def square_mask_multi_channel(x, mask_ratio=0.6,mask_size=20):
    """
    用很多个小方格遮住图片
    """
    for i_channel in range(0, x.shape[0]):
        if random.random() >= paint_prob:
            continue
        if random.random() < in_paint_prob:
            x[i_channel] = image_in_painting_2d(x[i_channel])
        else:
            x[i_channel] = image_out_painting_2d(x[i_channel])
    return x
def image_in_out_painting_2d_multi_channel(x,paint_prob,in_paint_prob):
    for i_channel in range(0, x.shape[0]):
        if random.random() >= paint_prob:
            continue
        if random.random() < in_paint_prob:
            x[i_channel] = image_in_painting_2d(x[i_channel])
        else:
            x[i_channel] = image_out_painting_2d(x[i_channel])
    return x
# TODO 挖去某些channel的，然后还原。层数不用挖，其他都可以挖
# TODO 改变层数标志，根据这一层的预测临近层的。可以预测pet，ct，organ

def image_in_painting_2d(x):
    img_rows, img_cols = x.shape
    cnt = 6
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

        x[noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x,
                                                               block_noise_size_y, ) * 1.0
        cnt -= 1
    return x

def image_out_painting_2d(x):
    img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1],) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)

    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)

    x[noise_x:noise_x+block_noise_size_x,
      noise_y:noise_y+block_noise_size_y,] = image_temp[noise_x:noise_x+block_noise_size_x,
                                                       noise_y:noise_y+block_noise_size_y,]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

        x[noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y] = image_temp[noise_x:noise_x+block_noise_size_x,
                                                           noise_y:noise_y+block_noise_size_y]
        cnt -= 1
    return x


def image_out_painting_2d_multi_channel(x, prob):
    for i_channel in range(0, x.shape[0]):
        if random.random() >= prob:
            continue
        x[i_channel] = image_out_painting_2d(x[i_channel])
    return x

def remove_channel(x, prob=0.1):
    # 移除某些通道，通过其他通道去复原
    for i_channel in range(0, x.shape[0]):
        if random.random() <= prob:
            x[i_channel] = np.zeros((x[i_channel].shape)) #TODO 0 or rand?
    return x

def cross_channel_replace_once(x, x1,prob=0.3):
    img_rows, img_cols = x.shape
    cnt = 3
    while cnt > 0 and random.random() < 1:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)

        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)

        x[noise_x:noise_x+block_noise_size_x,
          noise_y:noise_y+block_noise_size_y] = x1[noise_x:noise_x+block_noise_size_x,
                                                    noise_y:noise_y+block_noise_size_y]
        cnt -= 1
    return x
def cross_channel_replace(x, prob=0.3):
    channels = list(range(len(x)))
    cnt = len(x)
    while cnt > 0 and random.random() < prob:
        c_pair = random.sample(channels, 2)
        x[c_pair[0]] = cross_channel_replace_once(x[c_pair[0]], x[c_pair[1]])
        cnt -= 1
    return x

def generate_pair_2d(img, batch_size, config, status="test"):
    img_rows, img_cols = img.shape[2], img.shape[3]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            x[n], y[n] = data_augmentation_2d(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling_2d(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting_2d(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting_2d(x[n])

        # # Save sample images module
        # if config.save_samples is not None and status == "train" and random.random() < 0.01:
        #     n_sample = random.choice( [i for i in range(config.batch_size)] )
        #     sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
        #     sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
        #     sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
        #     sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
        #     final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        #     final_sample = final_sample * 255.0
        #     final_sample = final_sample.astype(np.uint8)
        #     file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
        #     imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (x, y)
def blur(x,prob=0.5):
    blur_x = copy.deepcopy(x)
    for i_channel in range(x.shape[0]):
        if random.random() >= prob:
            continue
        else:
            aug = iaa.GaussianBlur(sigma=(1,3))
            blur_x[i_channel] = aug.augment_image(x[i_channel])
    return blur_x
class LocalPixelShuffling():
    def __init__(self,img_size,template_num):
        self.img_size=img_size
        self.templates = self.get_template(template_num,img_size)
    def get_template(self,num,size):
        templates = []
        for i in range(num):
            a = np.linspace(0,size**2-1,num=size**2,endpoint=True,dtype=np.int)
            a = a.reshape((size,size))
            a = local_pixel_shuffling_2d(a)
            a = a.astype(np.int)
            templates.append(a.flatten())
        return templates

    def proc(self,img,prob=0.5):
        rand_idx = random.randint(0,len(self.templates)-1)
        order = self.templates[rand_idx]

        for i_channel in range(img.shape[0]):
            if random.random() >= prob:
                continue
            a = img[i_channel].flatten()
            a[order] = a
            img[i_channel] = a.reshape((self.img_size,self.img_size))

        return img

def generate_pair_multi_channel(img, config, status="test",lps=None):
    # img_rows, img_cols = img.shape[2], img.shape[3]

    y = img
    # Autoencoder
    x = copy.deepcopy(y)
    # Flip
    # x, y = data_augmentation_2d_multi_channel(x, y, config.flip_rate)

    #移动像素，除了层数以外，pet,ct,ct_stack，organ可以有，PET_stack待验证，可以某些c有，有些c没有
    # Local Shuffle Pixel
    x = lps.proc(x, prob=config.local_rate)

    # x = local_pixel_shuffling_2d_multi_channel(x, prob=config.local_rate)
    # x = blur(x,config.local_rate)
    # Apply non-Linear transformation with an assigned probability
    #TODO ct，ct_stack，都可以有，层数不用，organ，pet，petstack待验证（因为是亮度相关的）

    x = nonlinear_transformation_multi_channel(x, config.nonlinear_rate)

    # Inpainting & Outpainting
    #挖去某些channel的，然后还原。层数不用挖，其他都可以挖
    x = image_in_out_painting_2d_multi_channel(x, config.paint_rate,
                                           config.inpaint_rate)

    #TODO inpaint outpaint替换为随机采样小方框。（参考的masked autoencoder）


    # TODO 改变层数标志，根据这一层的预测临近层的。可以预测pet，ct，organ
    # if random.random() < config.other_layer_rate:

    #考虑直接去掉某些层？
    x = remove_channel(x,config.remove_channel_rate)

    #将一个通道的一块复制到另一通道 （试了不太行）
    # x = cross_channel_replace(x,config.cross_channel_replace_rate)
    # Save sample images module
    # final_sample = np.zeros((320,320))
    # for i_channel in range(x.shape[0]):
    #     temp = x[i_channel]
    #     mint = np.min(temp)
    #     maxt = np.max(temp)
    #     temp = 255 * (temp-mint)/(maxt - mint)
    #     final_sample = np.concatenate((final_sample, temp),axis=0)
    #
    # final_sample = final_sample.astype(np.uint8)
    # file_name = str(random.randint(0,100)) + '.png'
    # imageio.imwrite(os.path.join(r'/PCa_Segmentation/cache/ModelGenesis/result/pic', file_name), final_sample)

    return x, y