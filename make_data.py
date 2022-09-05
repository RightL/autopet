import pandas

print("11")
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
from utils_data_proc import *
def nii2h5():
    """区分有病灶的和没病灶的，保存成csv"""
    data_path = r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions'
    have_lesion = []
    no_lesion = []
    for folder1 in os.listdir(data_path):
        for folder2 in os.listdir(os.path.join(data_path,folder1)):
            file_path = os.path.join(folder1,folder2)

            mask_nii = sitk.ReadImage(os.path.join(data_path,file_path,'SEG.nii'))
            mask_nii = sitk.GetArrayFromImage(mask_nii)
            print(mask_nii.shape,file_path)
            if np.sum(mask_nii)==0:
                no_lesion.append(file_path)
                print('no',file_path)
            else:
                have_lesion.append(file_path)
                print('have',file_path)


    csv = pd.DataFrame(data=have_lesion)
    csv.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/have_lesion.csv', header='have_lesion', index=None)

    csv = pd.DataFrame(data=no_lesion)
    csv.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/no_lesion.csv', header='no_lesion', index=None)

def have_lesion_or_not():
    """区分有病灶的和没病灶的，保存成csv"""
    data_path = r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions'
    have_lesion = []
    no_lesion = []
    for folder1 in os.listdir(data_path):
        for folder2 in os.listdir(os.path.join(data_path,folder1)):
            file_path = os.path.join(folder1,folder2)
            mask_nii = sitk.ReadImage(os.path.join(data_path,file_path,'SEG.nii'))
            mask_nii = sitk.GetArrayFromImage(mask_nii)
            print(mask_nii.shape,file_path)
    #         if np.sum(mask_nii)==0:
    #             no_lesion.append(file_path)
    #             print('no',file_path)
    #         else:
    #             have_lesion.append(file_path)
    #             print('have',file_path)
    #
    #
    # csv = pd.DataFrame(data=have_lesion)
    # csv.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/have_lesion.csv', header='have_lesion', index=None)
    #
    # csv = pd.DataFrame(data=no_lesion)
    # csv.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/no_lesion.csv', header='no_lesion', index=None)

def new_train_split():
    path_and_id = pandas.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/path_and_id.csv')
    lesion_list = []
    patient_max_len = []
    for patient_id in range(len(path_and_id)):
        record = []
        nii_path = os.path.join(r'/data/newnas/ZSN/2022_miccai_petct/data/FDG-PET-CT-Lesions/',
                            path_and_id['path'][int(patient_id)])
        mask_nii = sitk.ReadImage(os.path.join(nii_path,'SEG.nii.gz'))
        mask = sitk.GetArrayFromImage(mask_nii)
        patient_max_len.append(mask.shape[0])
        if mask.max()==0:
            continue
        cca = sitk.ConnectedComponentImageFilter()
        cca.SetFullyConnected(False)
        # cca.FullyConnectedOff()
        _input = sitk.GetImageFromArray(mask.astype(np.uint8))
        output_ex = cca.Execute(_input)
        lss_filter = sitk.LabelShapeStatisticsImageFilter()
        lss_filter.Execute(output_ex)
        num = cca.GetObjectCount()

        for i in range(1, num + 1):
            centroid = lss_filter.GetCentroid(i)

            assert int(centroid[2])<mask.shape[0]
            add = True
            for j in record:
                if abs(j-int(centroid[2])) < 8:
                    add=False
            if add:
                lesion_list.append(str(patient_id)+'.'+str(int(centroid[2]))+'.h5')
                print(str(patient_id)+'.'+str(int(centroid[2]))+'.h5')
                record.append(int(centroid[2]))
    train_slice = pd.DataFrame(data=lesion_list)
    train_slice.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/sparce_lesion_h5.csv', header='sp_lesion', index=None)
    train_slice = pd.DataFrame(data=patient_max_len)
    train_slice.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/patient_max_len.csv', header='max_len', index=None)
    tpl = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_patient_lesion.csv')
    tpl=list(tpl['pid'])
    lesion_list = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/sparce_lesion_h5.csv')['spl'].tolist()
    train_slice = []
    for id in tpl:
        for h5_fname in lesion_list:
            if str(id) == h5_fname.split('.')[0]:
                train_slice.append(h5_fname)
    train_slice = pd.DataFrame(data=train_slice)
    train_slice.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_sparce_lesion_h5.csv', header='t5', index=None)

def train_test_split():
    csv = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/patient_have_lesion.csv')
    patient_have_lesion = csv[csv['have_lesion']==1]
    patient_no_lesion = csv[csv['have_lesion']==0]

    test_have = patient_have_lesion.sample(n=5)
    train_have = patient_have_lesion.drop(labels=test_have.index)
    val_have = train_have.sample(n=20)
    train_have = train_have.drop(labels=val_have.index)

    test_no = patient_no_lesion.sample(n=5)
    train_no = patient_no_lesion.drop(labels=test_no.index)
    val_no = train_no.sample(n=20)
    train_no = train_no.drop(labels=val_no.index)

    #病灶图像路径
    h5_fname_csv = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/h5fname_lesion_slice.csv')
    train_slice = []
    val_lesion_slice = []
    for id in train_have['patient_id']:
        for h5_fname in h5_fname_csv['ID']:
            if str(id) == h5_fname.split('.')[0]:
                train_slice.append(h5_fname)
    for id in val_have['patient_id']:
        for h5_fname in h5_fname_csv['ID']:
            if str(id) == h5_fname.split('.')[0]:
                val_lesion_slice.append(h5_fname)

    train_slice = pd.DataFrame(data=train_slice)
    train_slice.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_lesion_img.csv', header=None, index=None)

    #无病灶图像路径
    h5_fname_csv = pd.read_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/h5fname_healthy_slice.csv')
    train_slice = []
    val_healthy_slice = []
    for id in train_no['patient_id']:
        for h5_fname in h5_fname_csv['ID']:
            if str(id) == h5_fname.split('.')[0]:
                train_slice.append(h5_fname)
    for id in train_have['patient_id']:
        for h5_fname in h5_fname_csv['ID']:
            if str(id) == h5_fname.split('.')[0]:
                train_slice.append(h5_fname)

    for id in val_no['patient_id']:
        for h5_fname in h5_fname_csv['ID']:
            if str(id) == h5_fname.split('.')[0]:
                val_healthy_slice.append(h5_fname)
    train_slice = pd.DataFrame(data=train_slice)
    train_slice.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_no_img.csv', header=None, index=None)



    #验证集一半有病灶，一半无病灶
    val_lesion_slice = pd.DataFrame(data=val_lesion_slice)
    val_healthy_slice = pd.DataFrame(data=val_healthy_slice)
    val_healthy_slice = val_healthy_slice.sample(n=len(val_lesion_slice))
    val_all = pd.concat([val_lesion_slice,val_healthy_slice])
    val_all.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/val_img.csv', header=None, index=None)


    train_have.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_patient_lesion.csv', header=None, index=None)
    train_no.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/train_patient_healthy.csv', header=None, index=None)
    test_have.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/test_patient_lesion.csv', header=None, index=None)
    test_no.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/test_patient_healthy.csv', header=None, index=None)
    val_have.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/val_patient_lesion.csv', header=None, index=None)
    val_no.to_csv(r'/data/newnas/ZSN/2022_miccai_petct/data/val_patient_healthy.csv', header=None, index=None)
def get_stack_layer(filename):
    # 先生成h5，再获取stack
    save_folder = r'/data/newnas/ZSN/2022_miccai_petct/data/h5_data/v1/'
    z_len = 9

    patient_id = filename.split('.')[0]
    layer = filename.split('.')[1]
    pet_all = None
    ct_all = None
    mask_all = None
    for iz in range(int(1 - (z_len + 1) / 2), int((z_len + 1) / 2)):
        h5_fname = save_folder + patient_id + '.' + str(int(layer) + iz) + '.h5'
        if os.path.exists(h5_fname):
            h5_data = h5py.File(h5_fname, 'r')
            pet = h5_data['PET'][()]
            ct = h5_data['CT'][()]
            mask = h5_data['mask'][()].astype(np.uint8)
            h5_data.close()
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

        pet_all = concat(pet_all, pet)
        ct_all = concat(ct_all, ct)
        mask_all = concat(mask_all, mask)

    pet_stack_num_layer = 9
    pet_stack = get_pet_stack(pet_all, pet_stack_num_layer)
    ct_emerge_stack, ct_disappear_stack = get_ct_stack(ct_all, 9)

    return pet_stack,ct_emerge_stack,ct_disappear_stack

def get_stack():
    #先生成h5，再获取stack
    save_folder = r'/data/newnas/ZSN/2022_miccai_petct/data/h5_data/v1/'
    all_files_list = os.listdir(save_folder)
    z_len=9
    for filename in all_files_list[2*len(all_files_list)//3:]:
        patient_id = filename.split('.')[0]
        layer = filename.split('.')[1]
        pet_all = None
        ct_all = None
        mask_all = None
        for iz in range(int(1-(z_len+1)/2), int((z_len+1)/2)):
            h5_fname = save_folder+patient_id+'.'+str(int(layer)+iz)+'.h5'
            if os.path.exists(h5_fname):
                h5_data = h5py.File(h5_fname, 'r')
                pet = h5_data['PET'][()]
                ct = h5_data['CT'][()]
                mask = h5_data['mask'][()].astype(np.uint8)
                h5_data.close()
                if pet.shape[0]!=400:
                    pet = cv2.resize(pet,(400,400))
                if ct.shape[0]!=400:
                    ct = cv2.resize(ct,(400,400))
                if mask.shape[0]!=400:
                    mask = cv2.resize(mask,(400,400))
            else:
                pet= np.zeros((400,400))
                ct = np.zeros((400, 400))
                mask = np.zeros((400, 400))


            pet_all = concat(pet_all,pet)
            ct_all = concat(ct_all, ct)
            mask_all = concat(mask_all, mask)


        pet_stack_num_layer = 9
        pet_stack = get_pet_stack(pet_all, pet_stack_num_layer)
        ct_emerge_stack, ct_disappear_stack = get_ct_stack(ct_all, 9)
        h5_data = h5py.File(os.path.join(save_folder,filename), 'a')
        if 'pet_stack' in h5_data.keys():
            del h5_data['pet_stack']
        if 'ct_emerge_stack' in h5_data.keys():
            del h5_data['ct_emerge_stack']
        if 'ct_disappear_stack' in h5_data.keys():
            del h5_data['ct_disappear_stack']

        h5_data['pet_stack']=pet_stack
        h5_data['ct_emerge_stack'] = ct_emerge_stack
        h5_data['ct_disappear_stack'] = ct_disappear_stack
        h5_data.close()
        print(filename)

def change_type():
    save_folder = r'/data/newnas/ZSN/2022_miccai_petct/data/h5_data/v1/'
    all_files_list = os.listdir(save_folder)
    z_len=9
    for filename in all_files_list:
        h5_data = h5py.File(os.path.join(save_folder,filename), 'a')
        ct_emerge_stack = h5_data['ct_emerge_stack'][()]
        ct_disappear_stack = h5_data['ct_disappear_stack'][()]
        pet = h5_data['PET'][()]
        mask = h5_data['mask'][()].astype(np.uint8)
        # if 'pet_stack' in h5_data.keys():
        #     del h5_data['pet_stack']
        # else:
        #     pet_stack,ct_emerge_stack,ct_disappear_stack=get_stack_layer(filename)
        # if 'ct_emerge_stack' in h5_data.keys():
        #     del h5_data['ct_emerge_stack']
        # else:
        #     pet_stack,ct_emerge_stack,ct_disappear_stack=get_stack_layer(filename)
        # if 'ct_disappear_stack' in h5_data.keys():
        #     del h5_data['ct_disappear_stack']
        # else:
        #     pet_stack,ct_emerge_stack,ct_disappear_stack=get_stack_layer(filename)

        del h5_data['PET']
        del h5_data['mask']
        del h5_data['ct_disappear_stack']
        del h5_data['ct_emerge_stack']
        h5_data['PET'] = pet.astype(np.float16)

        h5_data['mask'] = mask.astype(np.uint8)
        h5_data['ct_emerge_stack'] = ct_emerge_stack.astype(np.float16)
        h5_data['ct_disappear_stack'] = ct_disappear_stack.astype(np.float16)
        h5_data.close()
        print(filename)



# change_type()
# get_stack()
new_train_split()