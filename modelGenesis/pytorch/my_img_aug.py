import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa


def data_aug_multimod(imgs_all, masks_all):
    """
    输入3维图像和标签（不够维数的需要在最后面补维）,返回进行了相同扩增的图像和标签,输入输出格式为numpy
    imgs_all的通道数代表了有多少模态；
    masks_all是所有标签类型的数据，其最后一个通道是GT
    """
    # 设定扩增方法
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 设定随机函数
    seq = iaa.Sequential(
        [
            sometimes(iaa.Crop(percent=(0, 0.1))),  # 对随机的一部分图像做crop操作 crop的幅度为0到10%

            sometimes(iaa.Affine(  # 对一部分图像做仿射变换
                scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},  # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
                rotate=(-30, 30),  # 旋转±45度之间
                shear=(-10, 10),  # 剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],  # 使用最邻近差值或者双线性差值
                cval=(0, 255),
                mode=ia.ALL,  # 边缘填充,随机0到255间的值
            )),

        ],
        random_order=True  # 随机的顺序把这些操作用在图像上
    )
    seq_det = seq.to_deterministic()  # 确定一个数据增强的序列

    # ============== 图像增强处理 =====================
    # 对图像的扩增
    mod_num = imgs_all.shape[2]
    for i in range(mod_num):
        # 标准化输入格式
        img = np.array(imgs_all[:, :, i])
        # 线性变化到0-255
        img_max = np.max(img)
        img_min = np.min(img)
        if (img_max - img_min) == 0:  # pet是全0层，ct是全为-360的层，空白层线性变化会导致出现nan
            img_aug = img  # 0就不需要做扩增了
        else:
            img = (img - img_min) / (img_max - img_min) * 255
            # 做扩增
            img_aug = seq_det.augment_image(img)
            # 变化会原来数值范围
            img_aug = img_aug / 255 * (img_max - img_min) + img_min
        # 放回imgs_all
        imgs_all[:, :, i] = img_aug
    # 对label的扩增
    label_num = masks_all.shape[2]
    for i in range(label_num):
        mask = np.array(masks_all[:, :, i])
        # 线性变化到0-255
        mask_min = np.min(mask)
        mask_max = np.max(mask)
        if mask_max - mask_min == 0:  # 无病灶层，防止空白层导致nan
            mask_aug = (mask - mask_min).astype(np.float32)
        else:
            mask = ((mask - mask_min) / (mask_max - mask_min) * 255).astype(np.uint8)
            # 分割标签格式
            segmap = ia.SegmentationMapsOnImage(mask, shape=mask.shape)
            # 将方法应用在分割标签上，并且转换回np类型
            mask_aug = seq_det.augment_segmentation_maps(segmap)
            mask_aug = mask_aug.get_arr()  # 有时会是bool型，大部分时候是uint8，如果是bool则不需要更改范围
            # 原来数值范围
            if mask_aug.dtype == 'bool':
                mask_aug = mask_aug.astype(np.float32)
            else:
                mask_aug = mask_aug / 255 * (mask_max - mask_min) + mask_min
        # 放回masks_all
        masks_all[:, :, i] = mask_aug

    return imgs_all, masks_all
