import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys

import torch


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_data()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()

    return false_pos
def false_pos_pix_gpu(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    gt_array = torch.tensor(gt_array).cuda()
    pred_conn_comp = torch.tensor(pred_conn_comp.astype(np.int64)).cuda()
    # pred_array = torch.tensor(pred_array).cuda()
    # false_pos = 0
    false_pos = torch.tensor(0,dtype=torch.float32).cuda()
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = torch.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    false_pos = false_pos.cpu().numpy()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg


def false_neg_pix_gpu(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    false_neg = torch.tensor(0,dtype=torch.float32).cuda()
    # gt_array=torch.tensor(gt_array).cuda()
    gt_conn_comp = torch.tensor(gt_conn_comp.astype(np.int64)).cuda()
    pred_array = torch.tensor(pred_array).cuda()
    for idx in range(1, gt_conn_comp.max() + 1):
        comp_mask = torch.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg = false_neg + comp_mask.sum()
    false_neg = false_neg.cpu().numpy()
    return false_neg

def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score

def dice_score_gpu(mask1,mask2):
    # compute foreground Dice coefficient
    mask1 = torch.tensor(mask1).cuda()
    mask2 = torch.tensor(mask2).cuda()
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score



def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)
    # print(((gt_array.astype(np.int16)-pred_array.astype(np.int16))**2).sum())
    false_neg_vol = false_neg_pix_gpu(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix_gpu(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score_gpu(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol
def compute_metrics_gpu(nii_gt_path, nii_pred_path):
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)
    # print(((gt_array.astype(np.int16)-pred_array.astype(np.int16))**2).sum())
    false_neg_vol = false_neg_pix_gpu(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix_gpu(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score_gpu(gt_array,pred_array)


if __name__ == "__main__":

    nii_gt_path, nii_pred_path = sys.argv

    nii_gt_path = plb.Path(nii_gt_path)
    nii_pred_path = plb.Path(nii_pred_path)
    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)

    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    csv_rows = [nii_gt_path.name,dice_sc, false_pos_vol, false_neg_vol]

    with open("metrics.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header) 
        writer.writerows(csv_rows)





