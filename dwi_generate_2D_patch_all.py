import scipy.io as sio
import sys
sys.path.extend('../')
from utils.sphere_interp import *
from dipy.io import read_bvals_bvecs
import nibabel as nib
import pathlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from dipy.io.image import load_nifti
from scipy.ndimage import rotate
def normalize_data(dMRI_data, mask, bvals):
    """Normalize data."""
    """
        Arguments:
            dMRI_data (4D numpy array): diffusion MRI data
            mask (3D numpy array): brain mask
            bvals (1D numpy array): bvalues

        Returns:
            (4D numpy array): normalized diffusion MRI data
    """
    dMRI_data_n = np.zeros_like(dMRI_data)

    idx_0 = np.where(bvals < 50)[0]
    b0_mean = np.mean(dMRI_data[:, :, :, idx_0], axis=3)

    dMRI_norm = (b0_mean[mask == 1] + 1.e-12)
    dMRI_data_n[mask == 1, :] = (dMRI_data[mask == 1, :].T / dMRI_norm).T

    dMRI_data_n_clip = np.clip(dMRI_data_n, 0., 1.)

    return dMRI_data_n_clip

index_1 = [1, 6, 7, 9, 11, 12, 14, 19, 20, 21, 23, 26, 28, 32, 35, 40, 41, 42, 46, 47, 48, 50, 54, 56, 58, 66, 71, 72,
           82, 84]
index_2 = [1, 2, 4, 7, 9, 12, 16, 17, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 41, 42, 43, 44, 51, 52, 55, 61, 79, 85,
           86, 89]
index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54, 61, 63,
           64, 70]

save_train_path = '/mnt/disk2/xth/HCPData_2D/NODDI_40_180_Nb0/train/'
save_val_path = '/mnt/disk2/xth/HCPData_2D/NODDI_40_180_Nb0/val/'

root = '/mnt/disk1/100-HCP/'
hcp_noddi_path = '/mnt/disk1/xth/NODDI_LABELS/'
hcp_dti_path = '/home/thx/Code/DTI_LABELS_b1000_full/'
dwi_90_sh = []
dwi_90 = []

files = sorted(os.listdir(root))
s = 0
for file in files:
    if s <= 29:
        print(file)
        file_str = str(file)
        file_name = file_str.split("_")[0]
        # if file_name == '102008':
        #     continue
        # if file_name == '109830':
        #     break
        data_path = root  + '/' + file_name + '/T1w/Diffusion/data.nii.gz'
        bval_path = root  + '/' + file_name + '/T1w/Diffusion/bvals'
        bvec_path = root  + '/' + file_name + '/T1w/Diffusion/bvecs'
        mask_path = root  + '/' + file_name + '/T1w/Diffusion/nodif_brain_mask.nii.gz'

        # data.shape (145, 174, 145, 288)
        data, affine = load_nifti(data_path)
        # bvals = 0, 1000, 2000, 3000; bvecs.shape (288, 3)
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        mask, _ = load_nifti(mask_path)

        data = normalize_data(data, mask, bvals)

        # choose only b=0,1000 shells
        # ind_1000 = (bvals > 900) * (bvals < 1100) + (bvals < 100)
        ind_1000 = (bvals > 900) * (bvals < 1100)
        bvecs_1000 = bvecs[ind_1000, :]
        dwi_1000 = data[:, :, :, ind_1000]
        # print('dwi_1000 shape:', dwi_1000.shape)
        bvals_1000 = bvals[ind_1000]
        dwi_1000 = dwi_1000 * mask[..., np.newaxis]
        dwi_1000_30 = dwi_1000[:, :, :, index_1]

        bval_1000_30 = bvals_1000[index_2]
        bvecs_1000_30 = bvecs_1000[index_2, :]

        ind_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_2000 = bvecs[ind_2000,:]
        dwi_2000 = data[:, :, :, ind_2000]
        bvals_2000 = bvals[ind_2000]
        dwi_2000 = dwi_2000 * mask[..., np.newaxis]
        dwi_2000_30 = dwi_2000[:, :, :, index_2]

        data = np.concatenate((dwi_1000, dwi_2000), axis=3)
        # print('data shape:', data.shape)
        # data = normalize(data, mask)
        print('data_train shape=', data.shape)

        icvf_path = hcp_noddi_path + '/' + file_name + '/FIT_ICVF.nii.gz'
        isovf_path = hcp_noddi_path + '/' + file_name + '/FIT_ISOVF.nii.gz'
        od_path = hcp_noddi_path + '/' + file_name + '/FIT_OD.nii.gz'
        mask_path = root  + '/' + file_name + '/T1w/Diffusion/nodif_brain_mask.nii.gz'

        GT_mask, _ = load_nifti(mask_path)

        # (145, 174, 145)
        GT_ICVF, _ = load_nifti(icvf_path)
        GT_ICVF[np.isnan(GT_ICVF)] = 0
        GT_ICVF = np.clip(GT_ICVF, 0, 1)
        GT_ICVF = GT_ICVF[..., np.newaxis]
        print('GT_ICVF shape:', GT_ICVF.shape)

        GT_ISOVF, _ = load_nifti(isovf_path)
        GT_ISOVF[np.isnan(GT_ISOVF)] = 0
        GT_ISOVF = np.clip(GT_ISOVF, 0, 1)
        GT_ISOVF = GT_ISOVF[..., np.newaxis]

        GT_OD, _ = load_nifti(od_path)
        GT_OD[np.isnan(GT_OD)] = 0
        GT_OD = np.clip(GT_OD, 0, 1)
        GT_OD = GT_OD[..., np.newaxis]

        n = 0
        # print('data_shape3', data.shape[2])
        for k in range(data.shape[2]):
            input_patch = data[:, :, k, :]

            filename = f"input_{file_name}_{n}.nii.gz"
            nib.save(nib.Nifti1Image(input_patch, np.eye(4)), os.path.join(save_train_path, filename))
            n = n + 1
        print('input_patch shape:', input_patch.shape)
        print('input_n=', n)
        n = 0

        for kk in range(GT_ICVF.shape[2]):

            icvf_patch = GT_ICVF[:, :, kk, :]
            isovf_patch = GT_ISOVF[:, :, kk, :]
            od_patch = GT_OD[:, :, kk, :]
            output_patch = np.concatenate((icvf_patch, isovf_patch, od_patch), axis=2)

            filename = f"output_{file_name}_{n}.nii.gz"
            nib.save(nib.Nifti1Image(output_patch, np.eye(4)), os.path.join(save_train_path, filename))
            n = n + 1

        print('output_patch shape:', output_patch.shape)
        print('output_n=', n)
        s = s + 1
        print('train_s=', s)

    elif 29 < s <= 32:
        print('exiting the program.')
        sys.exit()
        print(file)
        file_str = str(file)
        file_name = file_str.split("_")[0]
        # if file_name == '102008':
        #     continue
        # if file_name == '109830':
        #     break
        data_path = root + '/' + file_name + '/T1w/Diffusion/data.nii.gz'
        bval_path = root + '/' + file_name + '/T1w/Diffusion/bvals'
        bvec_path = root + '/' + file_name + '/T1w/Diffusion/bvecs'
        mask_path = root + '/' + file_name + '/T1w/Diffusion/nodif_brain_mask.nii.gz'

        # data.shape (145, 174, 145, 288)
        data, affine = load_nifti(data_path)
        # bvals = 0, 1000, 2000, 3000; bvecs.shape (288, 3)
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        mask, _ = load_nifti(mask_path)
        data = normalize_data(data, mask, bvals)

        ind_1000 = (bvals > 900) * (bvals < 1100)
        bvecs_1000 = bvecs[ind_1000, :]
        dwi_1000 = data[:, :, :, ind_1000]
        # print('dwi_1000 shape:', dwi_1000.shape)
        # bvals_1000 = bvals[index_1]
        dwi_1000 = dwi_1000 * mask[..., np.newaxis]
        dwi_1000_30 = dwi_1000[:, :, :, index_1]

        ind_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_2000 = bvecs[ind_2000, :]
        dwi_2000 = data[:, :, :, ind_2000]
        bvals_2000 = bvals[ind_2000]
        dwi_2000 = dwi_2000 * mask[..., np.newaxis]
        dwi_2000_30 = dwi_2000[:, :, :, index_2]

        data = np.concatenate((dwi_1000, dwi_2000), axis=3)
        data = normalize(data, mask)
        print('data_val shape=', data.shape)

        icvf_path = hcp_noddi_path + '/' + file_name + '/FIT_ICVF.nii.gz'
        isovf_path = hcp_noddi_path + '/' + file_name + '/FIT_ISOVF.nii.gz'
        od_path = hcp_noddi_path + '/' + file_name + '/FIT_OD.nii.gz'
        mask_path = root  + '/' + file_name + '/T1w/Diffusion/nodif_brain_mask.nii.gz'

        GT_mask, _ = load_nifti(mask_path)

        # (145, 174, 145)
        GT_ICVF, _ = load_nifti(icvf_path)
        GT_ICVF[np.isnan(GT_ICVF)] = 0
        GT_ICVF = np.clip(GT_ICVF, 0, 1)
        GT_ICVF = GT_ICVF[..., np.newaxis]
        print('GT_ICVF shape:', GT_ICVF.shape)

        GT_ISOVF, _ = load_nifti(isovf_path)
        GT_ISOVF[np.isnan(GT_ISOVF)] = 0
        GT_ISOVF = np.clip(GT_ISOVF, 0, 1)
        GT_ISOVF = GT_ISOVF[..., np.newaxis]

        GT_OD, _ = load_nifti(od_path)
        GT_OD[np.isnan(GT_OD)] = 0
        GT_OD = np.clip(GT_OD, 0, 1)
        GT_OD = GT_OD[..., np.newaxis]
        n = 0
        for k in range(data.shape[2]):
            input_patch = data[:, :, k, :]

            filename = f"input_{file_name}_{n}.nii.gz"
            nib.save(nib.Nifti1Image(input_patch, np.eye(4)), os.path.join(save_val_path, filename))
            n = n + 1
        print('input_patch shape:', input_patch.shape)
        print('input_n=', n)
        n = 0

        for kk in range(GT_ICVF.shape[2]):

            icvf_patch = GT_ICVF[:, :, kk, :]
            isovf_patch = GT_ISOVF[:, :, kk, :]
            od_patch = GT_OD[:, :, kk, :]
            output_patch = np.concatenate((icvf_patch, isovf_patch, od_patch), axis=2)

            filename = f"output_{file_name}_{n}.nii.gz"
            nib.save(nib.Nifti1Image(output_patch, np.eye(4)), os.path.join(save_val_path, filename))
            n = n + 1
        print('output_patch shape:', output_patch.shape)
        print('output_n=', n)
        s = s + 1
        print('val_s=', s)
    else:
        print('End')
        s = s + 1
        print('test_s=', s)

