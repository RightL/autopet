import SimpleITK as sitk
import numpy as np
import skimage as ski
from skimage import measure


def get_fold_filelist_from_file(fold=1):
    # 获取内部交叉验证训练集、验证集以及外部测试集的id
    txt_train_5fold = r'/PCa_Segmentation/data/new-data-0622/train_in&valid_in.txt'
    lines = []
    train_ids = []
    with open(txt_train_5fold, 'r') as f:
        for line in f.readlines():
            lines.append(line[0:-1])  # 去掉最后的\n
    for i in range(1, 20, 2):
        train_ids.append(lines[i].split(','))
    # 自定义分折情况
    Kfold_train_valid_test = {0: {'train': train_ids[0], 'val': train_ids[1]},
                              1: {'train': train_ids[2], 'val': train_ids[3]},
                              2: {'train': train_ids[4], 'val': train_ids[5]},
                              3: {'train': train_ids[6], 'val': train_ids[7]},
                              4: {'train': train_ids[8], 'val': train_ids[9]}}

    # 获取想要的分折的结果
    train_id = np.sort(np.array(Kfold_train_valid_test[fold - 1]['train'], np.uint16))
    val_id = np.sort(np.array(Kfold_train_valid_test[fold - 1]['val'], np.uint16))

    # 测试集
    txt_train_test = r'/PCa_Segmentation/data/new-data-0622/train&test.txt'
    lines = []
    with open(txt_train_test, 'r') as f:
        for line in f.readlines():
            lines.append(line[0:-1])  # 去掉最后的\n
    test_id = np.sort(np.array(lines[3].split(','), np.uint16))

    return train_id, val_id, test_id


def getDSC(SR, GT):
    """
    3维计算DSC，输入都是二值图，格式是array
    """
    Inter = np.sum(((SR + GT) == 2).astype(np.float32))
    DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)
    return DC


def get_detection_result(SR, GT, detect_threshold=0.5):
    """
    3维计算检测指标，precision/recall/f1-score
    其中precision同PPV，recall同sensitivity，在检测中无法计算specificity
    :param:detect_threshold:重合程度多少认为是正确检测到
    """
    tp1, fp1, fn1 = 0, 0, 0  # tn在检测中是无意义的
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(SR.astype(np.uint8))
    output_ex = cca.Execute(_input)
    pre_labeled = sitk.GetArrayFromImage(output_ex)
    num_pre = cca.GetObjectCount()
    # for ii in range(1, num_pre + 1):
    #     pre_one = pre_labeled == ii  # 取到其中一个连通域
    #     cover_area = pre_one * GT  # 和金标准的重合区域
    #     if np.sum(cover_area) / np.sum(pre_one) >= detect_threshold:  # 重合率大于阈值
    #         tp1 += 1
    #     else:
    #         fp1 += 1
    _input = sitk.GetImageFromArray(GT.astype(np.uint8))
    output_ex = cca.Execute(_input)
    gt_labeled = sitk.GetArrayFromImage(output_ex)
    num_gt = cca.GetObjectCount()
    for ii in range(1, num_gt + 1):
        gt_one = gt_labeled == ii  # 取到其中一个连通域
        cover_area = gt_one * SR  # 该连通域和预测结果的重合区域
        if np.sum(cover_area) / np.sum(gt_one) >= detect_threshold:  # 重合率大于阈值
            tp1 += 1
        else:
            fn1 += 1
    fp1 = num_pre - tp1
    if fp1 < 0:
        fp1 = 0
    # 得到每例结果
    precision1 = tp1 / (tp1 + fp1 + 1e-6)
    recall1 = tp1 / (tp1 + fn1 + 1e-6)
    f1_score1 = 2 * precision1 * recall1 / (precision1 + recall1 + 1e-6)
    return [precision1, recall1, f1_score1, tp1, fp1, fn1]


def check_componnent_3d(mask, pet=None, min_size=100, max_size=1e8, pet_min=1):
    """
    检查全身分割结果，去掉不符合的病灶
    """
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)

        if np.sum(label_temp[:]) <= min_size:  # 体积太小
            mask[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
        if np.sum(label_temp[:]) >= max_size:  # 体积太大
            mask[label_temp] = 0
            # print(np.sum(label_temp[:]))
            continue
        # 获取区域在z轴上的范围，如果不连续则去掉
        label_temp_z = np.sum(np.sum(label_temp, axis=2), axis=1)
        label_temp_z = label_temp_z > 0
        if np.sum(label_temp_z) <= 1:  # z轴上不连续
            mask[label_temp] = 0
            continue
        if pet is not None:
            if np.max(pet[label_temp][:]) <= pet_min:  # 均值太小
                mask[label_temp] = 0
                continue
    return mask


def check_componnent_2d(mask, min_size=10):
    # 检查连通域(2d)
    for j in range(mask.shape[0]):
        label_t, num_t = ski.measure.label(mask[j, :, :], connectivity=2, return_num=True)
        for jj in range(1, num_t + 1):
            label_temp_t = (label_t == jj)
            if np.sum(label_temp_t[:]) <= min_size:  # 体积太小
                mask[j, label_temp_t] = 0
    return mask


def check_componnent_organ(mask, organ_mask):
    """
    检查全身分割结果，对于前列腺癌，去掉在肾脏、胃部、脾内的
    """
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        if np.sum(label_temp[organ_mask == 2]) > 0 or np.sum(label_temp[organ_mask == 3]) > 0 or \
                np.sum(label_temp[organ_mask == 4]) > 0 or np.sum(label_temp[organ_mask == 7]) > 0:  #
            mask[label_temp] = 0
    return mask


def check_liver(mask, organ_mask, PET_array):
    """
       检查全身分割结果，去掉SUVmax小于肝脏SUVmax
    """
    liver_mask = organ_mask == 6
    liver_suv = np.percentile(PET_array[liver_mask], 75)        # 因为肝脏mask会被肾脏干扰，因此取一个75位数
    if liver_suv > 9:      # 防止太高的情况出现
        liver_suv = 9
    # 检查连通域(3d)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask.astype(np.uint8))
    output_ex = cca.Execute(_input)
    label = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    for j in range(1, num + 1):
        label_temp = (label == j)
        if np.max(PET_array[label_temp]) < liver_suv:
            mask[label_temp] = 0
    return mask