#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri May 30 18:05:30 2025

@author: thx

CNN-Mamba 
'''
import os.path as osp
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.ndimage import rotate
from dataset_sh_x import DmriDataset, DmriDataset_uniform, DmriDataset_uniform_ns_noise, \
    DmriDataset_uniform_multi, DmriDataset_random_1, DmriDataset_random_2
# from dataset_sh_x_fft import DmriDataset, normalize
# from dataset_x import DmriDataset, DmriDataset_noise, DmriDataset_random
from model import *
from utils.train_utils import *
from utils.make_edge import *
from train_util import *
import pathlib
from dipy.io.image import load_nifti
import nibabel as nib


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

def normalize(imgs, mask):
    result = np.zeros(imgs.shape)

    for jj in np.arange(0, imgs.shape[-1]):
        for kk in np.arange(0, imgs.shape[-2]):
            img = imgs[:, :, kk, jj]
            nonzero_pos = np.nonzero(mask[:, :, kk])
            nonzero = img[nonzero_pos]
            if len(nonzero) != 0:
                min = np.min(nonzero)
                max = np.max(nonzero)
                result[:, :, kk, jj] = (img - min) / (max - min + 1e-8)
                blocks_false = [mask[:, :, kk] == 0]
                temp = np.where(blocks_false, 0, result[:, :, kk, jj])
                result[:, :, kk, jj] = np.reshape(temp, result[:, :, kk, jj].shape)
    return result

def build_model(model_cfg):
    if model_cfg['backbone']['type'] == 'ResNet':
        model = ResNet(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                       model_cfg['backbone']['out_channel'])
    else:
        print('No {} model'.format(model_cfg['backbone']['type']))
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predictive microstructure model")
    parser.add_argument("--config", default='./config/resnet_config.py',
                        type=str, help="Run the model config file address")
    # parser.add_argument("--test_subject_id", default=['672756', '677968', '680957', '685058', '111312',
    #                                                   '111514', '114823', '130518', '144226', '896879',
    #                                                   '177746', '185442', '195041', '200614', '204521',
    #                                                   '146129', '158035', '562345', '627549', '783462'],
    #                     type=list, help="Training file name")
    parser.add_argument("--data_path", default="/mnt/disk2/xth/HCPData_2D/test_20_180_Nb0/",
                        type=str, help="Name of the test data path")
    parser.add_argument("--data_root_path", default='/mnt/disk1/xth/Data_100/test/',
                        type=str, help="Name of the test data root path")
    parser.add_argument("--mask_name", default="nodif_brain_mask.nii.gz",
                        type=str, help="Name of the mask file for the data")
    parser.add_argument("--is_train", default=False,
                        type=bool, help="Name of the testing gold standard file")
    parser.add_argument("--is_generate_image", default=False,
                        type=bool, help="Whether to generate prediction")
    parser.add_argument("--generate_image_save_path", default='./parameter_ResNet_repeat/40subject_Nb0_sh_w1_1_w2_1_w3_0001/lr0.001/',
                        type=str, help="Whether to generate prediction")
    parser.add_argument("--microstructure_name", default="NODDI", choices=['NODDI', 'DKI'],
                        type=str, help="Name of the microstructure model")
    args = parser.parse_args()
    return args

def test(test_loader, model, test_start_time, device, model_name):
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test'):
        model.eval()

        test_input, test_label = data
        test_input = test_input.type(torch.FloatTensor)
        test_input = test_input.to(device)

        out = model(test_input)

        if i == 0:
            prediction = out.cpu().detach().numpy()
            gt = test_label.numpy()
        else:
            temp_out = out.cpu().detach().numpy()
            temp_gt = test_label.numpy()
            prediction = np.concatenate((prediction, temp_out), axis=0)
            gt = np.concatenate((gt, temp_gt), axis=0)
    test_time_cost = time.time() - test_start_time
    print(' took {:.2f} seconds'.format(test_time_cost))
    return prediction, gt


def main(args):
    device = torch.device("cuda:0")
    model_cfg, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)

    model = build_model(model_cfg)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    test_model_path = args.generate_image_save_path + '/test.pth'
    model.load_state_dict(torch.load(test_model_path, map_location='cuda:0'))
    print('Load parameters success!')

    test_all_start_time = time.time()
    files = sorted(os.listdir(args.data_path))

    for file in files:
        test_start_time = time.time()
        print(file)
        save_path = args.generate_image_save_path + '/test/' + f'{file}/'
        if os.path.exists(save_path):
            print("file has exist:", file)
        else:
            pathlib.Path(save_path).mkdir(exist_ok=True)

        test_data_path = args.data_path + f'{file}/'
        dataset_test = DmriDataset_uniform(test_data_path, 'test')

        test_loader = DataLoader(dataset_test, batch_size=data_cfg['batch_size'], shuffle=False,
                                  drop_last=False, num_workers=data_cfg['num_workers'])

        prediction, gt = test(test_loader, model, test_start_time, device, model_cfg['backbone']['type'])
        pred_whole_img = np.transpose(prediction, (1, 2, 0, 3))
        gt_whole_img = np.transpose(gt, (1, 2, 0, 3))
        print("max value:", np.max(pred_whole_img))
        print('min value:', np.min(pred_whole_img))
        print("max value:", np.max(gt_whole_img))
        print('min value:', np.min(gt_whole_img))

        print('pred_whole_img shape:', pred_whole_img.shape)
        print('gt_whole_img shape:', gt_whole_img.shape)
        # res = gt_whole_img - pred_whole_img
        save_pic = save_path + '/'
        prefix = 'epoch' + str(data_cfg['train']['epoches']) + '_lr' + str(optimizer_cfg['lr'])
        #
        # # icvf
        # plt.imshow(rotate(pred_whole_img[:, :, 75, 0], 90), cmap='gray')
        # # plt.title('test_icvf_pred')
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_ICVF_pred.png', bbox_inches='tight',
        #             pad_inches=0.0)  # 可设置dpi=300/600
        # plt.show()
        # plt.close()
        #
        # plt.imshow(rotate(gt_whole_img[:, :, 75, 0], 90), cmap='gray')
        # # plt.title('test_icvf_gt')
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_ICVF_gt.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # from matplotlib.colors import LinearSegmentedColormap
        #
        # # wb
        # colors = ["white", "blue"]
        # cmap_name = 'custom_colormap'
        # cm = LinearSegmentedColormap.from_list(cmap_name, colors)
        # plt.imshow(rotate(abs(res[:, :, 75, 0]), 90), clim=(0, 0.5), cmap=cm)
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_ICVF_res_wb_0.5.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # # isovf
        # plt.imshow(rotate(pred_whole_img[:, :, 75, 1], 90), cmap='gray')
        # # plt.title('test_isovf_pred')
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_ISOVF_pred.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # plt.imshow(rotate(gt_whole_img[:, :, 75, 1], 90), cmap='gray')
        # # plt.title('test_isovf_gt')
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_ISOVF_gt.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # # wb
        # colors = ["white", "blue"]
        # cmap_name = 'custom_colormap'
        # cm = LinearSegmentedColormap.from_list(cmap_name, colors)
        # plt.imshow(rotate(abs(res[:, :, 75, 1]), 90), clim=(0, 0.5), cmap=cm)
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_ISOVF_res_wb_0.5.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # # od
        # plt.imshow(rotate(pred_whole_img[:, :, 75, 2], 90), cmap='gray')
        # # plt.title('test_od_pred')
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_OD_pred.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # plt.imshow(rotate(gt_whole_img[:, :, 75, 2], 90), cmap='gray')
        # # plt.title('test_od_gt')
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_OD_gt.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()
        #
        # # wb
        # colors = ["white", "blue"]
        # cmap_name = 'custom_colormap'
        # cm = LinearSegmentedColormap.from_list(cmap_name, colors)
        # plt.imshow(rotate(abs(res[:, :, 75, 2]), 90), clim=(0, 0.5), cmap=cm)
        # plt.axis('off')
        # plt.savefig(save_pic + 'test_' + prefix + '_OD_res_wb_0.5.png', bbox_inches='tight', pad_inches=0.0)
        # plt.show()
        # plt.close()


        icvf_pred = pred_whole_img[:, :, :, 0]
        icvf_gt = gt_whole_img[:, :, :, 0]
        isovf_pred = pred_whole_img[:, :, :, 1]
        isovf_gt = gt_whole_img[:, :, :, 1]
        od_pred = pred_whole_img[:, :, :, 2]
        od_gt = gt_whole_img[:, :, :, 2]

        mse_icvf, rmse_icvf, psnr_icvf, nrmse_icvf, ssim_icvf = get_metric(
            icvf_pred.reshape(icvf_pred.shape + (1,)), icvf_gt.reshape(icvf_gt.shape + (1,)))
        print("Prediction metric: mse_icvf={}, rmse_icvf ={}, psnr_icvf={}, nrmse_icvf={}, ssim_icvf={}.".format(
            mse_icvf, rmse_icvf, psnr_icvf, nrmse_icvf, ssim_icvf))


        mse_isovf, rmse_isovf, psnr_isovf, nrmse_isovf, ssim_isovf = get_metric(
            isovf_pred.reshape(isovf_pred.shape + (1,)), isovf_gt.reshape(isovf_gt.shape + (1,)))
        print("Prediction metric: mse_isovf={}, rmse_isovf={}, psnr_isovf={}, nrmse_isovf={}, ssim_isovf={}.".format(
            mse_isovf, rmse_isovf, psnr_isovf, nrmse_isovf, ssim_isovf))

        mse_od, rmse_od, psnr_od, nrmse_od, ssim_od = get_metric(od_pred.reshape(od_pred.shape + (1,)),
                                                                       od_gt.reshape(od_gt.shape + (1,)))
        print("Prediction metric: mse_od={}, rmse_od={}, psnr_od={}, nrmse_od={}, ssim_od={}.".format(
            mse_od, rmse_od, psnr_od, nrmse_od, ssim_od))


        mse, rmse, psnr, nrmse, ssim = get_metric(pred_whole_img, gt_whole_img)
        print("Prediction metric: mse={}, rmse={}, psnr={}, nrmse={}, ssim={}.".format(mse, rmse, psnr, nrmse, ssim))
        metrics = {
            'index': ['icvf', 'isovf', 'od', 'mean'],
            'MSE': [mse_icvf, mse_isovf, mse_od, mse],
            'RMSE': [rmse_icvf, rmse_isovf, rmse_od, rmse],
            'NRMSE': [nrmse_icvf, nrmse_isovf, nrmse_od, nrmse],
            'PSNR': [psnr_icvf, psnr_isovf, psnr_od, psnr],
            'SSIM': [ssim_icvf, ssim_isovf, ssim_od, ssim]}
        df = pd.DataFrame(metrics)
        df = df.round(5)
        excel_file = save_pic + 'test_' + prefix + 'metrics.xlsx'
        df.to_excel(excel_file, index=False)

    test_all_time_cost = time.time() - test_all_start_time
    print(' took {:.2f} seconds'.format(test_all_time_cost))

if __name__ == "__main__":
    args = parse_args()
    main(args)
