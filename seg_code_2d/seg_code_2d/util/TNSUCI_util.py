import csv
import os
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import SimpleITK as sitk
warnings.filterwarnings('ignore')
sep = os.sep
filesep = sep  # 设置分隔符


def char_color(s, front=50, word=32):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    new_char = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"
    return new_char


def array_shuffle(x, axis=0, random_state=2020):
    """
    对多维度数组，在任意轴打乱顺序
    :param x: ndarray
    :param axis: 打乱的轴
    :return:打乱后的数组
    """
    new_index = list(range(x.shape[axis]))
    random.seed(random_state)
    random.shuffle(new_index)
    x_new = np.transpose(x, ([axis] + [i for i in list(range(len(x.shape))) if i is not axis]))
    x_new = x_new[new_index][:]
    new_dim = list(np.array(range(axis)) + 1) + [0] + list(np.array(range(len(x.shape) - axis - 1)) + axis + 1)
    x_new = np.transpose(x_new, tuple(new_dim))
    return x_new


def get_filelist_frompath(filepath, expname, sample_id=None):
    """
    读取文件夹中带有固定扩展名的文件
    :param filepath:
    :param expname: 扩展名，如'h5','PNG'
    :param sample_id: 可以只读取固定患者id的图片
    :return: 文件路径list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.' + expname):
                id = int(file.split('.')[0])  # 以`.`为分隔符,然后第一个,也就得到id
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.' + expname):
                file_List.append(os.path.join(filepath, file))
    return file_List


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def get_train_fold():
    # 获取训练集的id
    txt_train_test = r'/data/nas/heyixue_group/PCa/new-data-0622/train_in&valid_in.txt'
    lines = []
    train_ids = []
    with open(txt_train_test, 'r') as f:
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
    return Kfold_train_valid_test


def get_fold_filelist_train_some(csv_file, K=5, fold=1, extract_num=1, random_state=2020):
    """
       获取训练集里的设定样本数量的全身h5_list
       :param csv_file: 带有ID、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param patient_num: 抽取训练集内几例的全身图像
       :return: train和test的h5_list
    """
    # 获取自定义的分折信息
    Kfold_train_valid_test = get_train_fold()

    # h5列表
    csvlines = readCsv(csv_file)
    # header = csvlines[0]
    # print('header', header)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    train_set = []
    train_id = np.array(Kfold_train_valid_test[fold - 1]['train'], np.uint16)
    # 从训练集内随机抽取extract_num个id
    np.random.seed(random_state)  # 保证可重复性
    extract_train_id = np.random.choice(train_id, extract_num, replace=False)
    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in extract_train_id:
            train_set.append(h5_file)
    print('whole_body train_id:' + str(extract_train_id))
    return train_set


def get_fold_filelist_train_all(csv_file, fold=1, extract_num=1):
    """
       在训练集里按设定例数划分，取全身h5_list，组成一个集合
       :param csv_file: 带有ID、size的文件
       :param fold: 返回第几折,从1开始
       :param extract_num: 抽取训练集内几例的全身图像
       :return: train的h5_list的集合
    """
    # 获取自定义的分折信息
    Kfold_train_valid_test = get_train_fold()

    # h5列表
    csvlines = readCsv(csv_file)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    train_id = np.array(Kfold_train_valid_test[fold - 1]['train'], np.uint16)

    train_num_limit = int(len(train_id)/extract_num)*extract_num       # 根据取的数目，限制训练集取的范围，即drop_last
    train_set_all = []
    for i in range(0, train_num_limit, extract_num):
        # 训练集内按间隔取
        extract_train_id = train_id[i:i+extract_num]
        train_set = []
        for h5_file in data_id:
            if int(h5_file.split('_')[0]) in extract_train_id:
                train_set.append(h5_file)
        train_set_all.append(train_set)
    return train_set_all


def get_fold_filelist_from_file(csv_file, fold=1):
    # 获取自定义的分折信息
    Kfold_train_valid_test = get_train_fold()

    csvlines = readCsv(csv_file)
    header = csvlines[0]
    # print('header', header)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    train_set = []
    val_set = []

    train_id = np.array(Kfold_train_valid_test[fold - 1]['train'], np.uint16)
    val_id = np.array(Kfold_train_valid_test[fold - 1]['val'], np.uint16)
    print('train_id:' + str(train_id) + '\nvalid_id:' + str(val_id))

    for h5_file in data_id:
        if int(h5_file.split('_')[0]) in val_id:
            val_set.append(h5_file)
        else:
            train_set.append(h5_file)

    return [train_set, val_set]


def get_fold_filelist_sn(csv_file, K=3, fold=1, random_state=2020):
    """
       利用Kfold获取分折结果
       :param csv_file: 带有ID、CATE、size的文件
       :param K: 分折折数
       :param fold: 返回第几折,从1开始
       :param random_state: 随机数种子
       :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
       :param validation_r: 抽取出验证集占训练集的比例
       :return: train和test的h5_list
       """
    csvlines = readCsv(csv_file)
    header = csvlines[0]
    print('header', header)
    nodules = csvlines[1:]
    data_id = [i[0] for i in nodules]

    patient_id = []
    for file in data_id:
        file_id = file.split("_")[0]
        patient_id.append(int(file_id))
    patient_num = list(set(patient_id))  # 按病人分折

    fold_train = []
    fold_test = []

    kf = KFold(n_splits=K, random_state=random_state, shuffle=True)
    for train_index, test_index in kf.split(patient_num):
        # print("TRAIN:", train_index, "TEST:", test_index)
        fold_train.append(train_index)
        fold_test.append(test_index)

    train_id = fold_train[fold - 1] + 1  # split是从0开始的,所以要加1
    test_id = fold_test[fold - 1] + 1
    print('train_id:' + str(train_id) + '\ntest_id:' + str(test_id))

    train_set = []
    test_set = []

    for h5_file in data_id:
        if str(h5_file.split('_')[0]) in str(test_id):
            test_set.append(h5_file)
        else:
            train_set.append(h5_file)

    return [train_set, test_set]


def print_logger(logger, savepth):
    for index, key in enumerate(logger.keys()):
        figg = plt.figure()
        plt.plot(logger[key])
        figg.savefig(savepth + sep + key + '.PNG')
        plt.close()


def save_nii(save_nii, CT_nii, save_path, save_mask=True):
    """
    保存nii
    :param save_nii: 需要保存的nii图像的array
    :param CT_nii: 配准的图像，用于获取同样的信息
    :param save_path: 保存路径
    :param save_mask: 保存的是否是mask，默认是True
    :return:
    """
    if save_mask:
        # 保存mask_nii
        save_sitk = sitk.GetImageFromArray(save_nii.astype(np.uint8))
        save_sitk.CopyInformation(CT_nii)
        save_sitk = sitk.Cast(save_sitk, sitk.sitkUInt8)
    else:
        # 保存img_nii
        save_sitk = sitk.GetImageFromArray(save_nii.astype(np.float))
        save_sitk.CopyInformation(CT_nii)
        save_sitk = sitk.Cast(save_sitk, sitk.sitkFloat32)

    sitk.WriteImage(save_sitk, save_path)
    print(save_path + ' processing successfully!')
