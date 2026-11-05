import os
import argparse
# from common.Loss_2d import Loss_2d
# from common.My_Loss import My_loss
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
# from common.NiiDataset_mix import NiiDataset_random, NiiDataset_uniform, NiiDataset_random_dwi_sh
# from common.NiiDataset import NiiDataset
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_root_mse as NRMSE

from scipy.ndimage import rotate
import pandas as pd
# from torch.optim.lr_scheduler import StepLR

# Some useful functions
def get_num_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def RMSE(pred, gt):
    return np.sqrt(MSE(pred, gt))


def nRMSE(a1, a_ref):
    n_ch = a_ref.shape[-1]
    b1 = np.reshape(a1, (-1, n_ch))
    b_ref = np.reshape(a_ref, (-1, n_ch))
    nrmse = np.zeros([1, n_ch])

    for n_c in range(n_ch):
        nrmse[0, n_c] = 2 * RMSE(b_ref[:, n_c], b1[:, n_c]) / RMSE(b_ref[:, n_c], -b_ref[:, n_c])
    nrmse = np.mean(nrmse)
    return nrmse


# pred (145, 174, 145, 3)
def get_metric(pred, gt):
    mse = MSE(pred, gt)
    rmse = RMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    # nrmse = nRMSE(pred, gt)
    nrmse = NRMSE(pred, gt, normalization='min-max')
    # ssim = SSIM(pred, gt)
    ssim = 0.0
    for j in range(pred.shape[-1]):
        ssim += SSIM(pred[:, :, :, j], gt[:, :, :, j], data_range=1)
    ssim = ssim / pred.shape[-1]

    return mse, rmse, psnr, nrmse, ssim


# pred (10, 128, 128, 66)
def get_22to1_metric(pred, gt):
    mse = MSE(pred, gt)
    rmse = RMSE(pred, gt)
    psnr = PSNR(pred, gt, data_range=1)
    ssim = 0.0
    for i in range(pred.shape[0]):
        ssim += SSIM(pred[i, :, :, :], gt[i, :, :, :])
    ssim = ssim / pred.shape[0]
    return mse, rmse, psnr, ssim

def get_scores_model(model, data_loader):
    metrics = []
    # pred_all = ()
    # gt_all = ()
    # toggle model to eval mode
    model.eval()
    # turn off gradients since they will not be used here
    i = 0
    with torch.no_grad():
        for x, y_tg in data_loader:
            i = i + 1
            x = x.cuda()
            y_tg = y_tg.cuda()
            y = model(x)
            pred = y.cpu().detach().numpy()
            # print('y_pred_shape:', y_pred.shape)
            tg = y_tg.cpu().detach().numpy()
            # print('y_tg_type:', type(y_tg))
            # print('y_tg_shape:', y_tg.shape)
            # if i==1:
            #     pred_all = y_pred
            #     # print('pred_all_1_shape:', pred_all.shape)
            #     gt_all = y_tg
            #     # print('gt_all_1_shape:', gt_all.shape)
            # else:
            #     pred_all = np.concatenate((pred_all, y_pred), axis=0)
            #     # print('pred_all_shape:', pred_all.shape)
            #     gt_all = np.concatenate((gt_all, y_tg), axis=0)
            #     # print('gt_all_shape:', gt_all.shape)
            loss1 = torch.mean(torch.pow((y - y_tg), 2))
            loss2 = torch.mean(torch.pow(y_tg, 2))
            if loss2.item() == 0:
                loss2 = 0.000001
            # print('loss2_type:', type(loss2))
            nmse = loss1 / loss2
            # print('nmse=', nmse)
            nmse = nmse.item()
            # print('nmse=', nmse)
            mse, rmse, psnr, ssim = get_1to1_metrics(pred, tg)
            metrics.append([mse, nmse, rmse, psnr, ssim])

    # results = np.concatenate(metrics, axis=0)
    return np.mean(metrics, axis=0)


def save_prediction(model, data_loader):
    # toggle model to eval mode
    model.eval()
    y_pred = np.zeros(0)
    y_gt = np.zeros(0)
    i = 0
    # turn off gradients since they will not be used here
    with torch.no_grad():
        for x, y_tg in data_loader:
            x = x.cuda()
            y = model(x)
            if i == 0:
                y_pred = y.cpu().detach().numpy()
                y_gt = y_tg.cpu().detach().numpy()
            else:
                y_pred = np.concatenate((y_pred, y.cpu().detach().numpy()), axis=0)
                y_gt = np.concatenate((y_gt, y_tg.cpu().detach().numpy()), axis=0)

            i += 1

    # file = h5py.File(save_path, 'w')
    # file['y_pred'] = y_pred
    return y_pred, y_gt


def get_score_model(model, data_loader):
    nrmse_list = []
    # toggle model to eval mode
    model.eval()
    # turn off gradients since they will not be used here
    with torch.no_grad():
        for x, y_tg in data_loader:
            x = x.cuda()
            y = model(x)
            this_nrmse = nRMSE(y.cpu().detach().numpy(), y_tg.cpu().detach().numpy())
            nrmse_list.append(this_nrmse)

    nrmse = np.concatenate(nrmse_list, axis=0)
    return np.mean(nrmse, axis=0)