#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri May 30 18:05:30 2025

@author: thx

CNN-Mamba 
'''
import os.path as osp
import os

import time

import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import structural_similarity_index_measure
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset_sh_x import DmriDataset, DmriDataset_uniform
# from dataset_sh_x_fft import DmriDataset, DmriDataset_uniform
from model import *
from utils import *
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def ssim_loss(y_true, y_pred):
    return 1 - structural_similarity_index_measure(y_pred, y_true, data_range=1.0)

def build_model(model_cfg):
    if model_cfg['backbone']['type'] == 'ResNet':
        model = ResNet(model_cfg['backbone']['in_channel'], model_cfg['backbone']['hidden'],
                       model_cfg['backbone']['out_channel'])
    else:
        print('No {} model'.format(model_cfg['backbone']['type']))
    return model

def build_optimizer(model, optimizer_cfg):
    if optimizer_cfg['type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg['lr'])
    elif optimizer_cfg['type'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=optimizer_cfg['lr'],
                                    momentum=optimizer_cfg['momentum'],
                                    weight_decay=optimizer_cfg['weight_decay'])
    else:
        print('no use {} optimizer'.format(optimizer_cfg['type']))
    return optimizer

def build_scheduler(lr_config, optimizer):
    if lr_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    else:
        print('no use {} scheduler'.format(lr_config['type']))
    return scheduler

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="predictive microstructure model")
    parser.add_argument("--config", default='./config/resnet_config.py',
                        type=str, help="Run the model config file address")
    # parser.add_argument("--valid_subject_id", default=['106319'],
    #                     type=list, help="Validing file name")
    parser.add_argument("--valid_subject_id", default=['121618', '123117', '124826'],
                        type=list, help="Validing file name")
    # parser.add_argument("--data_path_ori_dwi", default="/mnt/disk1/100-HCP/",
    #                     type=str, help="Name of the original dwi data path")
    parser.add_argument("--data_path_train", default="/mnt/disk2/xth/HCPData_2D/NODDI_40_180_Nb0/train/",
                        type=str, help="Name of the training data path")
    parser.add_argument("--data_path_val", default="/mnt/disk2/xth/HCPData_2D/NODDI_40_180_Nb0/val/",
                        type=str, help="Name of the validation data path")
    parser.add_argument("--is_train", default=True,
                        type=bool, help="Name of the testing gold standard file")
    parser.add_argument("--save_parameter_path", default="./parameter_ResNet_repeat/40subject_Nb0_sh_w1_1_w2_1_w3_0001/lr0.001_new/",
                        type=str, help="Name of the microstructure model")
    parser.add_argument("--microstructure_name", default="NODDI", choices=['NODDI', 'DKI'],
                        type=str, help="Name of the microstructure model")
    args = parser.parse_args()
    return args

def test(test_loader, model, test_start_time, subject_id, device, is_voxel, model_name, kwargs):

    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test'):
        model.eval()
        if model_name == 'GCNN':
            test_input = data.to(device)
            test_label = data.y.cpu()
        else:
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
    return restore_img(subject_id, gt, prediction, is_voxel, kwargs)


def train(train_loader_random, train_loader_uniform, model, optimizer, scheduler, device, is_voxel, model_name):
    total_loss = 0
    total_index1_loss = 0
    total_index2_loss = 0
    total_index3_loss = 0
    batch_number = 0
    epoch_time = time.time()

    for (x_in_random, y_tg_random), (x_in_uniform, y_tg_uniform) in tqdm(zip(train_loader_random, train_loader_uniform)):

    # for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='train'):
        model.train()
        batch_number += 1
        optimizer.zero_grad()

        input = torch.cat([x_in_random, x_in_uniform], dim=0)
        label_random = y_tg_random
        label_uniform = y_tg_uniform
        label_random = label_random.type(torch.FloatTensor)
        label_random = label_random.to(device)
        label_uniform = label_uniform.type(torch.FloatTensor)
        label_uniform = label_uniform.to(device)
        input = input.type(torch.FloatTensor)
        input = input.to(device)

        out = model(input)
        d = np.shape(out)[0]
        split_index = d // 2
        out_random = out[:split_index, :, :, :]
        out_uniform = out[-split_index:, :, :, :]

        #loss = 1*random_loss + 0.3*uniform_loss
        # loss = 1 * F.mse_loss(out_random, label_random) + 0.3 * F.mse_loss(out_uniform, label_uniform)
        # loss = 1*random_loss + 0.3*uniform_loss + 0.2*random_uniform
        # loss = 1 * F.mse_loss(out_random, label_random) + 1 * F.mse_loss(out_uniform, label_uniform) + \
        #        0.001 * F.mse_loss(out_random, out_uniform) + ssim_loss(out_random.permute(0, 3, 1, 2), label_random.permute(0, 3, 1, 2)) + \
        #        ssim_loss(out_uniform.permute(0, 3, 1, 2), label_uniform.permute(0, 3, 1, 2)) + \
        #        0.001 * ssim_loss(out_random.permute(0, 3, 1, 2), out_uniform.permute(0, 3, 1, 2))
        loss = 1 * F.mse_loss(out_random, label_random) + 1 * F.mse_loss(out_uniform, label_uniform) + \
               0.001 * F.mse_loss(out_random, out_uniform)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    total_loss = total_loss / batch_number
    epoch_time_cost = time.time() - epoch_time
    if scheduler is not None:
        scheduler.step()
    return total_loss, epoch_time_cost


def main(args):
    device = torch.device("cuda:1")
    model_cfg, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    os.makedirs(args.save_parameter_path, exist_ok=True)
    if not os.path.exists(args.save_parameter_path + '/' + model_cfg['backbone']['type']):
        os.mkdir(args.save_parameter_path + '/' + model_cfg['backbone']['type'])

    if data_cfg['train']['pretrained_flag']== True:
        best_model_path = os.path.join(args.save_parameter_path, 'pre_best_parameter.pth')
    else:
        best_model_path = os.path.join(args.save_parameter_path, 'best_parameter.pth')

    dataset_train_random = DmriDataset(args.data_path_train, data_type='train_val')
    sample_x = RandomSampler(range(len(dataset_train_random)), replacement=False)

    #or shuffle=False
    train_loader_random = DataLoader(dataset_train_random, batch_size=data_cfg['batch_size']//2, sampler=sample_x,
                              drop_last=True, num_workers=data_cfg['num_workers'])
    dataset_train_uniform = DmriDataset_uniform(args.data_path_train, data_type='train_val')
    train_loader_uniform = DataLoader(dataset_train_uniform, batch_size= data_cfg['batch_size']//2, sampler=sample_x,
                                     drop_last=True, num_workers=data_cfg['num_workers'])

    model = build_model(model_cfg)
    # model = nn.DataParallel(model, device_ids=[1, 2])
    model.to(device)

    optimizer = build_optimizer(model, optimizer_cfg)

    if data_cfg['train']['pretrained_flag'] == True:
        if os.path.isfile(data_cfg['train']['pretrained_weights']):
            model.load_state_dict(torch.load(data_cfg['train']['pretrained_weights']))
            print('Load parameters success!')
        else:
            print('Load parameters unsuccess!')

    if lr_config['is_scheduler'] == True:
        scheduler = build_scheduler(lr_config, optimizer)
    else:
        scheduler = None

    train_start_time = time.time()
    min_loss = 1000
    all_psnr_max = 0
    early_stop_number = 0
    for epoch in range(0, data_cfg['train']['epoches']):
        total_loss, epoch_time_cost = train(train_loader_random,
                                             train_loader_uniform,
                                             model,
                                             optimizer,
                                             scheduler,
                                             device,
                                             data_cfg['is_voxel'],
                                             model_cfg['backbone']['type'])
        if args.microstructure_name == 'NODDI':
            print('Epoch {} time {:.2f} seconds total_loss is: {} '.format(
                    epoch+ 1, epoch_time_cost, total_loss))

        if ((epoch + 1) % 1 == 0) and (total_loss < min_loss):  # % 1
            min_loss = total_loss
            print('min loss is {}'.format(min_loss))
            index1_psnr, index2_psnr, index3_psnr, all_psnr = 0, 0, 0, 0
            subject_number = len(args.valid_subject_id)
            for subject_id in args.valid_subject_id:
                test_start_time = time.time()
                subject_list = []
                subject_list.append(subject_id)
                val_data_path = args.data_path_val + f'{subject_id}/'
                dataset_val = DmriDataset_uniform(val_data_path, data_type='train_val')
                test_loader = DataLoader(dataset_val, batch_size=data_cfg['batch_size'], shuffle=True,
                                          drop_last=True, num_workers=data_cfg['num_workers'])

                if args.is_train:

                    index1_psnr_value, index2_psnr_value, index3_psnr_value, all_psnr_value = test(test_loader,
                                                                model, test_start_time, subject_id,
                                                                device, data_cfg['is_voxel'],
                                                                model_cfg['backbone']['type'], args)
                else:
                    print('not train')
                index1_psnr += index1_psnr_value
                index2_psnr += index2_psnr_value
                index3_psnr += index3_psnr_value
                all_psnr += all_psnr_value
            index1_psnr = index1_psnr / subject_number
            index2_psnr = index2_psnr / subject_number
            index3_psnr = index3_psnr / subject_number
            all_psnr = all_psnr / subject_number
            if args.microstructure_name == 'NODDI':
                print('valid suject mean psnr icvf is: {}  isovf is: {}  od is: {} all_noddi is: {} '.format(
                    index1_psnr, index2_psnr, index3_psnr, all_psnr))
            if args.microstructure_name == 'DKI':
                print('valid suject mean psnr ak is: {}  mk is: {}  rk is: {} all_dki is: {}'.format(
                    index1_psnr, index2_psnr, index3_psnr, all_psnr))
            if all_psnr_max < all_psnr:
                all_psnr_max = all_psnr
                torch.save(model.state_dict(), best_model_path)
                early_stop_number = 0
                print('{} model parameters have been stored!'.format(epoch))
            else:
                early_stop_number += 1
                print('break number is {}'.format(early_stop_number))
        else:
            early_stop_number += 1
            print('break number is {}'.format(early_stop_number))
        if early_stop_number > 9:
            print("\n============================")
            print("Main Training Converged")
            print("============================")

            # ==========================================
            # save the model from the last epoch of th current stage
            # ==========================================

            current_lr = optimizer.param_groups[0]['lr']

            last_stage_model = os.path.join(
                args.save_parameter_path,
                f"stage_lr_{current_lr:.0e}.pth"
            )

            torch.save(
                model.state_dict(),
                last_stage_model
            )

            print("Saved:", last_stage_model)

            # ==========================================
            # multi-stage refinement
            # ==========================================

            while True:

                #you can modify the learning rate decay factors as needed
                next_lr = current_lr * 0.1

                if next_lr < 1e-8:
                    print("\nRefinement Finished.")
                    print(f"Stop because lr={next_lr:.1e} < 1e-6")
                    break

                print("\n--------------------------------")
                print(f"LR decay: {current_lr:.1e} -> {next_lr:.1e}")
                print("--------------------------------")

                # ======================================
                # load the model from the previous stage
                # ======================================

                print("Load:", last_stage_model)

                checkpoint = torch.load(
                    last_stage_model,
                    map_location=device
                )

                model.load_state_dict(checkpoint)

                # ======================================
                #adjust the learning rate
                # ======================================

                for param_group in optimizer.param_groups:
                    param_group['lr'] = next_lr

                # ======================================
                # train for only one epoch
                # ======================================

                refine_loss, refine_time = train(
                    train_loader_random,
                    train_loader_uniform,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    data_cfg['is_voxel'],
                    model_cfg['backbone']['type']
                )

                print(
                    f"Refinement Epoch Finished | "
                    f"lr={next_lr:.1e} | "
                    f"loss={refine_loss:.6f}"
                )

                # ======================================
                # save the current model
                # ======================================

                current_stage_model = os.path.join(
                    args.save_parameter_path,
                    f"stage_lr_{next_lr:.0e}.pth"
                )

                torch.save(
                    model.state_dict(),
                    current_stage_model
                )

                print("Saved:", current_stage_model)

                # ======================================
                # update the reference model
                # ======================================

                last_stage_model = current_stage_model

                current_lr = next_lr

            break
        epoch_save_path = os.path.join(args.save_parameter_path, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_save_path)

        train_time_cost = time.time() - train_start_time
        print(' took {:.2f} seconds'.format(train_time_cost))

if __name__ == "__main__":
    args = parse_args()
    main(args)
