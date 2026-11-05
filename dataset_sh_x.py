from torch.utils.data import Dataset
import torch
import os
import nibabel as nib
import numpy as np

from utils.sphere_interp import *
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti
import random

def get_spherical_harmonics_coefficients(dwi, bvecs, sh_order, smooth, first=False):
    """ Compute coefficients of the spherical harmonics basis.
    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    Returns
    -------
    sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number of
        coeffs depends on `sh_order`.
    """
    bvecs = np.asarray(bvecs)
    #     dwi_weights = dwi.get_data().astype("float32")
    dwi_weights = dwi
    # Assuming all directions are on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs)

    # Fit SH to signal
    sph_harm_basis = sph_harm_lookup.get('tournier07')
    Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
    data_sh = np.dot(dwi_weights, invB.T)

    return data_sh

class DmriDataset(Dataset):
    def __init__(self, data_dir,  data_type):
        if data_type == 'train_val':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)

        elif data_type == 'test':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-2]), int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)


    def __getitem__(self, index):
        sh_order = 6
        smooth = 0.006
        data_file = self.data_files[index]
        subject = int(data_file.split('.')[0].split('_')[-2])
        subject_path = '/mnt/disk1/100-HCP/' + f'{subject}/'
        bvec_path = subject_path + '/T1w/Diffusion/bvecs'
        bval_path = subject_path + '/T1w/Diffusion/bvals'
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        int_1000 = (bvals > 900) * (bvals < 1100)
        int_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_1000 = bvecs[int_1000, :]
        bvecs_2000 = bvecs[int_2000, :]

        data_load = nib.load(os.path.join(self.data_dir, data_file)).get_fdata()

        data_1000 = data_load[:, :, :90]
        data_2000 = data_load[:, :, -90:]
        b1_index = data_1000.shape[-1]
        dirs_1 = []
        for i in range(b1_index):
            dirs_1.append(i)
        # dirs_1 = np.arange(0, 90, 1)
        b1_selected = random.sample(dirs_1, 30)
        b1_selected = sorted(b1_selected)
        dwi_1000_30 = data_1000[:, :, b1_selected]
        bvecs_1000_30 = bvecs_1000[b1_selected, :]
        data_sh_1000 = get_spherical_harmonics_coefficients(dwi_1000_30, bvecs_1000_30,
                                                            sh_order=sh_order, smooth=smooth)
        b2_index = data_2000.shape[-1]
        dirs_2 = []
        for i in range(b2_index):
            dirs_2.append(i)
        # dirs_1 = np.arange(0, 90, 1)
        b2_selected = random.sample(dirs_2, 30)
        b2_selected = sorted(b2_selected)
        dwi_2000_30 = data_2000[:, :, b2_selected]
        bvecs_2000_30 = bvecs_2000[b2_selected, :]
        data_sh_2000 = get_spherical_harmonics_coefficients(dwi_2000_30, bvecs_2000_30,
                                                            sh_order=sh_order, smooth=smooth)
        data = np.concatenate((data_sh_1000, data_sh_2000), axis=2)
        # data = np.concatenate((data, dwi_1000_30, dwi_2000_30), axis=2)
        # print('data shape:', data.shape)

        label_file = self.label_files[index]
        label = nib.load(os.path.join(self.data_dir, label_file)).get_fdata()
        x = torch.FloatTensor(data)
        y = torch.FloatTensor(label)

        return x, y

    def __len__(self):
        return len(self.data_files)

class DmriDataset_uniform(Dataset):
    def __init__(self, data_dir,  data_type):
        if data_type == 'train_val':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)

        elif data_type == 'test':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-2]), int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)


    def __getitem__(self, index):
        index_1 = [1, 6, 7, 9, 11, 12, 14, 19, 20, 21, 23, 26, 28, 32, 35, 40, 41, 42, 46, 47, 48, 50, 54, 56, 58, 66,
                   71, 72, 82, 84]
        index_2 = [1, 2, 4, 7, 9, 12, 16, 17, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 41, 42, 43, 44, 51, 52, 55, 61,
                   79, 85, 86, 89]
        # index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
        #            61, 63, 64, 70]

        sh_order = 6
        smooth = 0.006
        data_file = self.data_files[index]
        subject = int(data_file.split('.')[0].split('_')[-2])
        subject_path = '/mnt/disk1/100-HCP/' + f'{subject}/'
        bvec_path = subject_path + '/T1w/Diffusion/bvecs'
        bval_path = subject_path + '/T1w/Diffusion/bvals'
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        int_1000 = (bvals > 900) * (bvals < 1100)
        int_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_1000 = bvecs[int_1000, :]
        bvecs_2000 = bvecs[int_2000, :]

        data_load = nib.load(os.path.join(self.data_dir, data_file)).get_fdata()
        data_1000 = data_load[:, :, :90]
        data_2000 = data_load[:, :, -90:]
        dwi_1000_30 = data_1000[:, :, index_1]
        bvecs_1000_30 = bvecs_1000[index_1, :]
        data_sh_1000 = get_spherical_harmonics_coefficients(dwi_1000_30, bvecs_1000_30,
                                                            sh_order=sh_order, smooth=smooth)
        dwi_2000_30 = data_2000[:, :, index_2]
        bvecs_2000_30 = bvecs_2000[index_2, :]
        data_sh_2000 = get_spherical_harmonics_coefficients(dwi_2000_30, bvecs_2000_30,
                                                            sh_order=sh_order, smooth=smooth)
        data = np.concatenate((data_sh_1000, data_sh_2000), axis=2)

        label_file = self.label_files[index]
        # print('data shape:', data.shape)
        label = nib.load(os.path.join(self.data_dir, label_file)).get_fdata()

        x = torch.FloatTensor(data)
        y = torch.FloatTensor(label)

        return x, y

    def __len__(self):
        return len(self.data_files)

class DmriDataset_uniform_ns_noise(Dataset):
    def __init__(self, data_dir,  data_type):
        if data_type == 'train_val':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('rician_random_0075_input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)

        elif data_type == 'test':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('rician_random_0075_input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-2]), int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)


    def __getitem__(self, index):
        # index_1_uniform = [1, 6, 7, 9, 11, 12, 14, 19, 20, 21, 23, 26, 28, 32, 35, 40, 41, 42, 46, 47, 48, 50, 54, 56, 58, 66,
        #            71, 72, 82, 84]
        # index_2_uniform = [1, 2, 4, 7, 9, 12, 16, 17, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 41, 42, 43, 44, 51, 52, 55, 61,
        #            79, 85, 86, 89]
        # # index_3_uniform = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
        # #            61, 63, 64, 70]
        index_1_random = [2, 4, 10, 14, 17, 18, 23, 24, 31, 33, 36, 38, 40, 42, 49, 54, 57, 58, 59, 64, 68, 72, 73, 77,
                          78, 81, 84, 85, 88, 89]
        index_2_random = [1, 4, 9, 13, 22, 23, 26, 35, 40, 41, 43, 44, 45, 46, 47, 53, 54, 59, 60, 62, 66, 71, 72, 73,
                          74, 78, 80, 81, 85, 87]
        sh_order = 6
        smooth = 0.006
        data_file = self.data_files[index]
        subject = int(data_file.split('.')[0].split('_')[-2])
        subject_path = '/mnt/disk1/100-HCP/' + f'{subject}/'
        bvec_path = subject_path + '/T1w/Diffusion/bvecs'
        bval_path = subject_path + '/T1w/Diffusion/bvals'
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        int_1000 = (bvals > 900) * (bvals < 1100)
        int_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_1000 = bvecs[int_1000, :]
        bvecs_2000 = bvecs[int_2000, :]

        data_load = nib.load(os.path.join(self.data_dir, data_file)).get_fdata()
        # print('data shape:', data.shape)
        data_1000_30 = data_load[:, :, :30]
        data_2000_30 = data_load[:, :, -30:]
        bvecs_1000_30 = bvecs_1000[index_1_random, :]
        data_sh_1000 = get_spherical_harmonics_coefficients(data_1000_30, bvecs_1000_30,
                                                            sh_order=sh_order, smooth=smooth)
        bvecs_2000_30 = bvecs_2000[index_2_random, :]
        data_sh_2000 = get_spherical_harmonics_coefficients(data_2000_30, bvecs_2000_30,
                                                            sh_order=sh_order, smooth=smooth)
        data = np.concatenate((data_sh_1000, data_sh_2000), axis=2)

        label_file = self.label_files[index]
        # print('data shape:', data.shape)
        label = nib.load(os.path.join(self.data_dir, label_file)).get_fdata()
        x = torch.FloatTensor(data)
        y = torch.FloatTensor(label)

        return x, y

    def __len__(self):
        return len(self.data_files)

class DmriDataset_random_1(Dataset):
    def __init__(self, data_dir,  data_type):
        if data_type == 'train_val':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)

        elif data_type == 'test':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):

                return int(filename.split('.')[0].split('_')[-2]), int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)


    def __getitem__(self, index):
        index_1 = [2, 3, 10, 12, 16, 18, 23, 24, 26, 31, 32, 36, 37, 40, 43, 44, 46, 48, 55, 59, 60, 64, 66, 70, 76, 79,
                   80, 83, 85, 87]
        index_2 = [1, 5, 6, 13, 15, 19, 20, 22, 27, 29, 30, 33, 35, 38, 41, 47, 49, 50, 52, 56, 58, 62, 63, 67, 68, 71,
                   78, 84, 86, 88]
        # index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
        #            61, 63, 64, 70]

        sh_order = 6
        smooth = 0.006
        data_file = self.data_files[index]
        subject = int(data_file.split('.')[0].split('_')[-2])
        subject_path = '/mnt/disk1/100-HCP/' + f'{subject}/'
        bvec_path = subject_path + '/T1w/Diffusion/bvecs'
        bval_path = subject_path + '/T1w/Diffusion/bvals'
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        int_1000 = (bvals > 900) * (bvals < 1100)
        int_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_1000 = bvecs[int_1000, :]
        bvecs_2000 = bvecs[int_2000, :]

        data_load = nib.load(os.path.join(self.data_dir, data_file)).get_fdata()
        # print('data shape:', data.shape)
        data_1000 = data_load[:, :, :90]
        data_2000 = data_load[:, :, -90:]
        dwi_1000_30 = data_1000[:, :, index_1]
        bvecs_1000_30 = bvecs_1000[index_1, :]
        data_sh_1000 = get_spherical_harmonics_coefficients(dwi_1000_30, bvecs_1000_30,
                                                            sh_order=sh_order, smooth=smooth)
        dwi_2000_30 = data_2000[:, :, index_2]
        bvecs_2000_30 = bvecs_2000[index_2, :]
        data_sh_2000 = get_spherical_harmonics_coefficients(dwi_2000_30, bvecs_2000_30,
                                                            sh_order=sh_order, smooth=smooth)
        data = np.concatenate((data_sh_1000, data_sh_2000), axis=2)

        label_file = self.label_files[index]
        # print('data shape:', data.shape)
        label = nib.load(os.path.join(self.data_dir, label_file)).get_fdata()

        x = torch.FloatTensor(data)
        y = torch.FloatTensor(label)

        return x, y

    def __len__(self):
        return len(self.data_files)

class DmriDataset_random_2(Dataset):
    def __init__(self, data_dir,  data_type):
        if data_type == 'train_val':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)

        elif data_type == 'test':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_noddi_para_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-2]), int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)


    def __getitem__(self, index):
        index_1 = [0, 4, 7, 8, 9, 11, 14, 17, 21, 25, 28, 30, 34, 39, 42, 45, 51, 53, 54, 57, 61, 65, 69, 72, 73, 74, 75,
                   77, 81, 82, 89]
        index_2 = [0, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16, 17, 18, 19, 26, 27, 30, 34, 36, 38, 40, 62, 63, 65, 66, 74,
                   75, 78, 79, 87]
        # index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
        #            61, 63, 64, 70]

        sh_order = 6
        smooth = 0.006
        data_file = self.data_files[index]
        subject = int(data_file.split('.')[0].split('_')[-2])
        subject_path = '/mnt/disk1/100-HCP/' + f'{subject}/'
        bvec_path = subject_path + '/T1w/Diffusion/bvecs'
        bval_path = subject_path + '/T1w/Diffusion/bvals'
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        int_1000 = (bvals > 900) * (bvals < 1100)
        int_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_1000 = bvecs[int_1000, :]
        bvecs_2000 = bvecs[int_2000, :]

        data_load = nib.load(os.path.join(self.data_dir, data_file)).get_fdata()
        # print('data shape:', data.shape)
        data_1000 = data_load[:, :, :90]
        data_2000 = data_load[:, :, -90:]
        dwi_1000_30 = data_1000[:, :, index_1]
        bvecs_1000_30 = bvecs_1000[index_1, :]
        data_sh_1000 = get_spherical_harmonics_coefficients(dwi_1000_30, bvecs_1000_30,
                                                            sh_order=sh_order, smooth=smooth)
        dwi_2000_30 = data_2000[:, :, index_2]
        bvecs_2000_30 = bvecs_2000[index_2, :]
        data_sh_2000 = get_spherical_harmonics_coefficients(dwi_2000_30, bvecs_2000_30,
                                                            sh_order=sh_order, smooth=smooth)
        data = np.concatenate((data_sh_1000, data_sh_2000), axis=2)

        label_file = self.label_files[index]
        # print('data shape:', data.shape)
        label = nib.load(os.path.join(self.data_dir, label_file)).get_fdata()

        x = torch.FloatTensor(data)
        y = torch.FloatTensor(label)

        return x, y

    def __len__(self):
        return len(self.data_files)

class DmriDataset_uniform_multi(Dataset):
    def __init__(self, data_dir,  data_type):
        if data_type == 'train_val':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)

        elif data_type == 'test':
            self.data_dir = data_dir

            self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
            print('data_files=', len(self.data_files))
            self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_')])
            print('label_files=', len(self.label_files))

            def get_number(filename):
                return int(filename.split('.')[0].split('_')[-2]), int(filename.split('.')[0].split('_')[-1])

            self.data_files.sort(key=get_number)
            self.label_files.sort(key=get_number)


    def __getitem__(self, index):
        # index_1 = [1, 6, 7, 9, 11, 12, 14, 19, 20, 21, 23, 26, 28, 32, 35, 40, 41, 42, 46, 47, 48, 50, 54, 56, 58, 66,
        #            71, 72, 82, 84]
        # index_2 = [1, 2, 4, 7, 9, 12, 16, 17, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 41, 42, 43, 44, 51, 52, 55, 61,
        #            79, 85, 86, 89]
        # index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
        #            61, 63, 64, 70]
        b1_selected = [1, 6, 7, 9, 11, 12, 14, 19, 20, 21, 23, 26, 28, 32, 35, 40, 41, 42, 46, 47, 48, 50, 56, 58, 66,
                   71, 72, 82, 84]
        b2_selected = [1, 2, 4, 7, 9, 12, 16, 17, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 41, 42, 43, 44, 51, 55, 61,
                   79, 85, 86, 89]
        # index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
        #            61, 63, 64, 70]
        sh_order = 6
        smooth = 0.006
        data_file = self.data_files[index]
        subject = int(data_file.split('.')[0].split('_')[-2])
        subject_path = '/mnt/disk1/100-HCP/' + f'{subject}/'
        bvec_path = subject_path + '/T1w/Diffusion/bvecs'
        bval_path = subject_path + '/T1w/Diffusion/bvals'
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        int_1000 = (bvals > 900) * (bvals < 1100)
        int_2000 = (bvals > 1900) * (bvals < 2100)
        bvecs_1000 = bvecs[int_1000, :]
        bvecs_2000 = bvecs[int_2000, :]

        data_load = nib.load(os.path.join(self.data_dir, data_file)).get_fdata()
        # print('data shape:', data.shape)
        data_1000 = data_load[:, :, :90]
        data_2000 = data_load[:, :, -90:]

        # b1_selected = random.sample(index_1, 29)
        # b1_selected = sorted(b1_selected)
        dwi_1000_29 = data_1000[:, :, b1_selected]
        bvecs_1000_29 = bvecs_1000[b1_selected, :]
        data_sh_1000 = get_spherical_harmonics_coefficients(dwi_1000_29, bvecs_1000_29,
                                                            sh_order=sh_order, smooth=smooth)

        # b2_selected = random.sample(index_2, 29)
        # b2_selected = sorted(b2_selected)
        dwi_2000_29 = data_2000[:, :, b2_selected]
        bvecs_2000_29 = bvecs_2000[b2_selected, :]
        data_sh_2000 = get_spherical_harmonics_coefficients(dwi_2000_29, bvecs_2000_29,
                                                            sh_order=sh_order, smooth=smooth)
        data = np.concatenate((data_sh_1000, data_sh_2000), axis=2)

        label_file = self.label_files[index]
        # print('data shape:', data.shape)
        label = nib.load(os.path.join(self.data_dir, label_file)).get_fdata()

        x = torch.FloatTensor(data)
        y = torch.FloatTensor(label)

        return x, y

    def __len__(self):
        return len(self.data_files)