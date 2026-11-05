import os
import sys
import time
import types
sys.path.insert(0,os.getcwd())
import importlib
import numpy as np
import nibabel as nib
import math
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random


def file2dict(filename):
    (path,file) = os.path.split(filename)
    abspath = os.path.abspath(os.path.expanduser(path))
    sys.path.insert(0,abspath)
    mod = importlib.import_module(file.split('.')[0])
    sys.path.pop(0)
    cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
                and not isinstance(value, types.ModuleType)
                and not isinstance(value, types.FunctionType)
                    }
    return cfg_dict.get('model_cfg'), cfg_dict.get('data_cfg'),cfg_dict.get('lr_config'),cfg_dict.get('optimizer_cfg')

# def get_concatenate_data(data_path, subject_id, file_name, file_gt_name):
#     data_list = []
#     gt_data_list = []
#     for file_name_id in subject_id:
#         print('Reading Reading hcp data sub_ID: '+ str(file_name_id))
#         data = np.load(os.path.join(data_path, str(file_name_id) + '/' + file_name))
#         gt_data = np.load(os.path.join(data_path, str(file_name_id) + '/' + file_gt_name))
#         data_list.append(data)
#         gt_data_list.append(gt_data)
#         print('Reading finished!')
#     for i in range(len(data_list)):
#         if i == 0 :
#             temporary_test_data =data_list[i]
#             print('InputPatch shape is {}'.format(temporary_test_data.shape))
#             temporary_gt_data =gt_data_list[i]
#             print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
#         else:
#             temporary_test_data = np.concatenate((temporary_test_data, data_list[i]), 0)
#             temporary_gt_data = np.concatenate((temporary_gt_data, gt_data_list[i]), 0)
#             print('InputPatch shape is {}'.format(temporary_test_data.shape))
#             print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
#     print('InputPatch shape is {}'.format(temporary_test_data.shape))
#     print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
#     return temporary_test_data, temporary_gt_data

def get_concatenate_data(data_path):
    index_1 = [1, 6, 7, 9, 11, 12, 14, 19, 20, 21, 23, 26, 28, 32, 35, 40, 41, 42, 46, 47, 48, 50, 54, 56, 58, 66,
               71, 72, 82, 84]
    index_2 = [1, 2, 4, 7, 9, 12, 16, 17, 21, 24, 25, 26, 28, 32, 34, 35, 37, 38, 41, 42, 43, 44, 51, 52, 55, 61,
               79, 85, 86, 89]
    # index_3 = [1, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20, 22, 23, 24, 25, 26, 28, 29, 31, 32, 35, 37, 49, 53, 54,
    #            61, 63, 64, 70]
    data_list = []
    gt_data_list = []
    data_dir = data_path
    data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('input_')])
    label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('output_')])
    def get_number(filename):
        return int(filename.split('.')[0].split('_')[-1])
    data_files.sort(key=get_number)
    label_files.sort(key=get_number)
    for i in range(len(data_files)):
        if i == 0 :
            data = nib.load(os.path.join(data_dir, data_files[i])).get_fdata()
            data_1000 = data[:, :, :90]
            data_2000 = data[:, :, -90:]
            #random sampling
            #b1000
            b1_index = data_1000.shape[-1]
            dirs_1 = []
            for i_1 in range(b1_index):
                dirs_1.append(i_1)
            # dirs_1 = np.arange(0, 90, 1)
            b1_selected = random.sample(dirs_1, 30)
            b1_selected = sorted(b1_selected)
            dwi_1000_30 = data_1000[:, :, b1_selected]
            #b2000
            b2_index = data_2000.shape[-1]
            dirs_2 = []
            for i_2 in range(b2_index):
                dirs_2.append(i_2)
            # dirs_1 = np.arange(0, 90, 1)
            b2_selected = random.sample(dirs_2, 30)
            b2_selected = sorted(b2_selected)
            dwi_2000_30 = data_2000[:, :, b2_selected]
            #uniform sampling
            # dwi_1000_30 = data_1000[:, :, index_1]
            # dwi_2000_30 = data_2000[:, :, index_2]
            data = np.concatenate((dwi_1000_30, dwi_2000_30), axis=2)
            temporary_test_data = np.reshape(data, [1, 145, 174, 60])
            # temporary_test_data = temporary_test_data[:,60:90, 70:100, :]
            # temporary_test_data = data
            print('InputPatch shape is {}'.format(temporary_test_data.shape))
            label = nib.load(os.path.join(data_dir, label_files[i])).get_fdata()
            temporary_gt_data = np.reshape(label, [1, 145, 174, 3])
            # temporary_gt_data = temporary_gt_data[:,60:90, 70:100, :]
            # temporary_gt_data = label
            print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
        else:
            data = nib.load(os.path.join(data_dir, data_files[i])).get_fdata()
            data_1000 = data[:, :, :90]
            data_2000 = data[:, :, -90:]
            # random sampling
            # b1000
            b1_index = data_1000.shape[-1]
            dirs_1 = []
            for j_1 in range(b1_index):
                dirs_1.append(j_1)
            # dirs_1 = np.arange(0, 90, 1)
            b1_selected = random.sample(dirs_1, 30)
            b1_selected = sorted(b1_selected)
            dwi_1000_30 = data_1000[:, :, b1_selected]
            # b2000
            b2_index = data_2000.shape[-1]
            dirs_2 = []
            for j_2 in range(b2_index):
                dirs_2.append(j_2)
            # dirs_1 = np.arange(0, 90, 1)
            b2_selected = random.sample(dirs_2, 30)
            b2_selected = sorted(b2_selected)
            dwi_2000_30 = data_2000[:, :, b2_selected]
            # uniform sampling
            # dwi_1000_30 = data_1000[:, :, index_1]
            # dwi_2000_30 = data_2000[:, :, index_2]
            data = np.concatenate((dwi_1000_30, dwi_2000_30), axis=2)
            data = np.reshape(data, [1, 145, 174, 60])
            # data = data[:,60:90, 70:100, :]
            temporary_test_data = np.concatenate((temporary_test_data, data), 0)
            label = nib.load(os.path.join(data_dir, label_files[i])).get_fdata()
            label = np.reshape(label, [1, 145, 174, 3])
            # label = label[:,60:90, 70:100, :]
            temporary_gt_data = np.concatenate((temporary_gt_data, label), 0)
            # print('InputPatch shape is {}'.format(temporary_test_data.shape))
            # print('TargetPatch shape is {}'.format(temporary_gt_data.shape))

    print('InputPatch shape is {}'.format(temporary_test_data.shape))
    print('TargetPatch shape is {}'.format(temporary_gt_data.shape))
    return temporary_test_data, temporary_gt_data

def computer_psnr_1D(gt_data, prediction_data, model_name, is_train):
    # gt_icvf = gt_data[:, :, :, 0]
    # pre_icvf = prediction_data[:, :, :, 0]
    gt_icvf = gt_data[:, 0]
    pre_icvf = prediction_data[:, 0]


    icvf_nrmse_value = nrmse(gt_icvf, pre_icvf)
    icvf_psnr_value = psnr(gt_icvf, pre_icvf, data_range=1)
    # icvf_ssim_value = ssim(gt_icvf, pre_icvf, data_range=1)

    # gt_isovf = gt_data[:, :, :, 1]
    # pre_isovf = prediction_data[:, :, :, 1]
    gt_isovf = gt_data[:, 1]
    pre_isovf = prediction_data[:, 1]

    isovf_nrmse_value = nrmse(gt_isovf, pre_isovf)
    isovf_psnr_value = psnr(gt_isovf, pre_isovf, data_range=1)
    # isovf_ssim_value = ssim(gt_isovf, pre_isovf, data_range=1)

    # gt_od = gt_data[:, :, :, 2]
    # pre_od = prediction_data[:, :, :, 2]
    gt_od = gt_data[:, 2]
    pre_od = prediction_data[:, 2]

    od_nrmse_value = nrmse(gt_od, pre_od)
    od_psnr_value = psnr(gt_od, pre_od, data_range=1)
    # od_ssim_value = ssim(gt_od, pre_od, data_range=1)


    # gt_all = gt_data[:, :, :, 0:3]
    # pre_all = prediction_data[:, :, :, 0:3]
    gt_all = gt_data[:, 0:3]
    pre_all = prediction_data[:, 0:3]

    all_noddi_nrmse_value = nrmse(gt_all, pre_all)
    all_noddi_psnr_value = psnr(gt_all, pre_all, data_range=1)
    # all_noddi_ssim_value = ssim(gt_all, pre_all, data_range=1, channel_axis=-1)
    print('all_NODDI')
    print('NRMSE value is {}'.format(all_noddi_nrmse_value))
    print('PSNR value is {}'.format(all_noddi_psnr_value))
    # print('SSIM value is {}'.format(all_noddi_ssim_value))
    if is_train:
        return icvf_psnr_value, isovf_psnr_value, od_psnr_value, all_noddi_psnr_value
    else:
        return icvf_psnr_value, isovf_psnr_value, od_psnr_value, all_noddi_psnr_value, icvf_nrmse_value, isovf_nrmse_value, od_nrmse_value, all_noddi_nrmse_value
               # icvf_ssim_value, isovf_ssim_value, od_ssim_value, all_noddi_ssim_value
def computer_psnr(gt_data, prediction_data, model_name, is_train):
    if model_name == 'NODDI':
        gt_icvf = gt_data[:, :, :, 0]
        pre_icvf = prediction_data[:, :, :, 0]
        # gt_icvf = gt_data[:, 0]
        # pre_icvf = prediction_data[:, 0]
        icvf_nrmse_value = nrmse(gt_icvf, pre_icvf)
        icvf_psnr_value = psnr(gt_icvf, pre_icvf, data_range=1)
        icvf_ssim_value = ssim(gt_icvf, pre_icvf, data_range=1)

        gt_isovf = gt_data[:, :, :, 1]
        pre_isovf = prediction_data[:, :, :, 1]
        # gt_isovf = gt_data[:, 1]
        # pre_isovf = prediction_data[:, 1]
        isovf_nrmse_value = nrmse(gt_isovf, pre_isovf)
        isovf_psnr_value = psnr(gt_isovf, pre_isovf, data_range=1)
        isovf_ssim_value = ssim(gt_isovf, pre_isovf, data_range=1)

        gt_od = gt_data[:, :, :, 2]
        pre_od = prediction_data[:, :, :, 2]
        # gt_od = gt_data[:, 2]
        # pre_od = prediction_data[:, 2]
        od_nrmse_value = nrmse(gt_od, pre_od)
        od_psnr_value = psnr(gt_od, pre_od, data_range=1)
        od_ssim_value = ssim(gt_od, pre_od, data_range=1)

        #DKI
        gt_ak = gt_data[:, :, :, 3]
        pre_ak = prediction_data[:, :, :, 3]
        ak_nrmse_value = nrmse(gt_ak, pre_od)
        ak_psnr_value = psnr(gt_ak, pre_ak, data_range=1)
        ak_ssim_value = ssim(gt_ak, pre_ak, data_range=1)

        gt_mk = gt_data[:, :, :, 4]
        pre_mk = prediction_data[:, :, :, 4]
        mk_nrmse_value = nrmse(gt_mk, pre_mk)
        mk_psnr_value = psnr(gt_mk, pre_mk, data_range=1)
        mk_ssim_value = ssim(gt_mk, pre_mk, data_range=1)

        gt_rk = gt_data[:, :, :, 5]
        pre_rk = prediction_data[:, :, :, 5]
        rk_nrmse_value = nrmse(gt_rk, pre_rk)
        rk_psnr_value = psnr(gt_rk, pre_rk, data_range=1)
        rk_ssim_value = ssim(gt_rk, pre_rk, data_range=1)

        gt_kfa = gt_data[:, :, :, 6]
        pre_kfa = prediction_data[:, :, :, 6]
        kfa_nrmse_value = nrmse(gt_kfa, pre_kfa)
        kfa_psnr_value = psnr(gt_kfa, pre_kfa, data_range=1)
        kfa_ssim_value = ssim(gt_kfa, pre_kfa, data_range=1)

        #SADDI
        # gt_de = gt_data[:, :, :, 3]
        # pre_de = prediction_data[:, :, :, 3]
        # # gt_od = gt_data[:, 3]
        # # pre_od = prediction_data[:, 3]
        # de_nrmse_value = nrmse(gt_de, pre_de)
        # de_psnr_value = psnr(gt_de, pre_de, data_range=1)
        # de_ssim_value = ssim(gt_de, pre_de, data_range=1)
        #
        # gt_din = gt_data[:, :, :, 4]
        # pre_din = prediction_data[:, :, :, 4]
        # # gt_fa = gt_data[:, 4]
        # # pre_fa = prediction_data[:, 4]
        # din_nrmse_value = nrmse(gt_din, pre_din)
        # din_psnr_value = psnr(gt_din, pre_din, data_range=1)
        # din_ssim_value = ssim(gt_din, pre_din, data_range=1)
        #
        # gt_fextra = gt_data[:, :, :, 5]
        # pre_fextra = prediction_data[:, :, :, 5]
        # # gt_md = gt_data[:, 5]
        # # pre_md = prediction_data[:, 5]
        # fextra_nrmse_value = nrmse(gt_fextra, pre_fextra)
        # fextra_psnr_value = psnr(gt_fextra, pre_fextra, data_range=1)
        # fextra_ssim_value = ssim(gt_fextra, pre_fextra, data_range=1)
        #
        # gt_fneurite = gt_data[:, :, :, 6]
        # pre_dfneurite = prediction_data[:, :, :, 6]
        # # gt_od = gt_data[:, 3]
        # # pre_od = prediction_data[:, 3]
        # fneurite_nrmse_value = nrmse(gt_fneurite, pre_dfneurite)
        # fneurite_psnr_value = psnr(gt_fneurite, pre_dfneurite, data_range=1)
        # fneurite_ssim_value = ssim(gt_fneurite, pre_dfneurite, data_range=1)
        #
        # gt_fsoma = gt_data[:, :, :, 7]
        # pre_fsoma = prediction_data[:, :, :, 7]
        # # gt_fa = gt_data[:, 4]
        # # pre_fa = prediction_data[:, 4]
        # fsoma_nrmse_value = nrmse(gt_fsoma, pre_fsoma)
        # fsoma_psnr_value = psnr(gt_fsoma, pre_fsoma, data_range=1)
        # fsoma_ssim_value = ssim(gt_fsoma, pre_fsoma, data_range=1)
        #
        # gt_rsoma = gt_data[:, :, :, 8]
        # pre_frsoma = prediction_data[:, :, :, 8]
        # # gt_md = gt_data[:, 5]
        # # pre_md = prediction_data[:, 5]
        # rsoma_nrmse_value = nrmse(gt_rsoma, pre_frsoma)
        # rsoma_psnr_value = psnr(gt_rsoma, pre_frsoma, data_range=1)
        # rsoma_ssim_value = ssim(gt_rsoma, pre_frsoma, data_range=1)


        gt_all = gt_data[:, :, :, 0:7]
        pre_all = prediction_data[:, :, :, 0:7]
        # gt_all = gt_data[:, 0:6]
        # pre_all = prediction_data[:, 0:6]

        all_nrmse_value = nrmse(gt_all, pre_all)
        all_psnr_value = psnr(gt_all, pre_all, data_range=1)
        # all_noddi_ssim_value = ssim(gt_all, pre_all, data_range=1, channel_axis=-1)
        print('all')
        print('NRMSE value is {}'.format(all_nrmse_value))
        print('PSNR value is {}'.format(all_psnr_value))
        # print('SSIM value is {}'.format(all_noddi_ssim_value))
        if is_train:
            return icvf_psnr_value, isovf_psnr_value, od_psnr_value, ak_psnr_value, mk_psnr_value, rk_psnr_value, kfa_psnr_value, all_psnr_value
        else:
            return icvf_psnr_value, isovf_psnr_value, od_psnr_value, ak_psnr_value, mk_psnr_value, rk_psnr_value, kfa_psnr_value, all_psnr_value, \
                icvf_nrmse_value, isovf_nrmse_value, od_nrmse_value, ak_nrmse_value, mk_nrmse_value, rk_nrmse_value, kfa_nrmse_value, all_nrmse_value
                   # icvf_ssim_value, isovf_ssim_value, od_ssim_value, all_noddi_ssim_value
        # if is_train:
        #     return icvf_psnr_value, isovf_psnr_value, od_psnr_value, de_psnr_value, din_psnr_value, fextra_psnr_value, fneurite_psnr_value, fsoma_psnr_value, rsoma_psnr_value, all_psnr_value
        # else:
        #     return icvf_psnr_value, isovf_psnr_value, od_psnr_value, de_psnr_value, din_psnr_value, fextra_psnr_value, fneurite_psnr_value, fsoma_psnr_value, rsoma_psnr_value, all_psnr_value, \
        #         icvf_nrmse_value, isovf_nrmse_value, od_nrmse_value, de_nrmse_value, din_nrmse_value, fextra_nrmse_value, fneurite_nrmse_value, fsoma_nrmse_value, rsoma_nrmse_value, all_nrmse_value
        #            # icvf_ssim_value, isovf_ssim_value, od_ssim_value, all_noddi_ssim_value
    if model_name == 'DKI':
        gt_ak = gt_data[:, 0]
        pre_ak = prediction_data[:, 0]

        ak_nrmse_value = nrmse(gt_ak, pre_ak)
        ak_psnr_value = psnr(gt_ak, pre_ak, data_range=3)
        ak_ssim_value = ssim(gt_ak, pre_ak, data_range=3)
        print('ak')
        print('NRMSE value is {}'.format(ak_nrmse_value))
        print('PSNR value is {}'.format(ak_psnr_value))
        print('SSIM value is {}'.format(ak_ssim_value))

        gt_mk = gt_data[:, 1]
        pre_mk = prediction_data[:, 1]

        mk_nrmse_value = nrmse(gt_mk, pre_mk)
        mk_psnr_value = psnr(gt_mk, pre_mk, data_range=3)
        mk_ssim_value = ssim(gt_mk, pre_mk, data_range=3)
        print('mk')
        print('NRMSE value is {}'.format(mk_nrmse_value))
        print('PSNR value is {}'.format(mk_psnr_value))
        print('SSIM value is {}'.format(mk_ssim_value))

        gt_rk = gt_data[:, 2]
        pre_rk = prediction_data[:, 2]

        rk_nrmse_value = nrmse(gt_rk, pre_rk)
        rk_psnr_value = psnr(gt_rk, pre_rk, data_range=3)
        rk_ssim_value = ssim(gt_rk, pre_rk, data_range=3)
        print('rk')
        print('NRMSE value is {}'.format(rk_nrmse_value))
        print('PSNR value is {}'.format(rk_psnr_value))
        print('SSIM value is {}'.format(rk_ssim_value))

        gt_dki_all = gt_data[:, 0:3]
        pre_dki_all = prediction_data[:, 0:3]

        all_dki_nrmse_value = nrmse(gt_dki_all, pre_dki_all)
        all_dki_psnr_value = psnr(gt_dki_all, pre_dki_all, data_range=3)
        all_dki_ssim_value = ssim(gt_dki_all, pre_dki_all, data_range=3,multichannel=True)
        print('All_DKI')
        print('NRMSE value is {}'.format(all_dki_nrmse_value))
        print('PSNR value is {}'.format(all_dki_psnr_value))
        print('SSIM value is {}'.format(all_dki_ssim_value))
        if is_train:
            return ak_psnr_value, mk_psnr_value, rk_psnr_value, all_dki_psnr_value
        else:
            return ak_psnr_value, mk_psnr_value, rk_psnr_value, all_dki_psnr_value, ak_nrmse_value, mk_nrmse_value, rk_nrmse_value, all_dki_nrmse_value, \
                   ak_ssim_value, mk_ssim_value, rk_ssim_value, all_dki_ssim_value

def restore_img(subject_id, gt, prediction, is_voxel, args):
    # mask_file = os.path.join(os.path.join(args.data_path, subject_id), args.mask_name)
    # mask = nib.load("/mnt/disk1/100-HCP/100307/T1w/Diffusion/nodif_brain_mask.nii.gz")
    # mask = mask.get_fdata()
    # x_size = mask.shape[0]
    # y_size = mask.shape[1]
    # z_size = mask.shape[2]
    # gt_like = np.zeros([x_size, y_size, z_size, 3])
    # prediction_like = np.zeros([x_size, y_size, z_size, 3])
    # psnr_gt = []
    # psnr_prediction = []
    #
    # if is_voxel:
    #     start = -1
    #     for xx in range(0, x_size, 1):
    #         for yy in range(0, y_size, 1):
    #             for zz in range(0, z_size, 1):
    #                 if (mask[xx, yy, zz] > 0):
    #                     start += 1
    #                     gt_like[xx, yy, zz, :] = gt[start, :]
    #                     prediction_like[xx, yy, zz, :] = prediction[start, :]
    #                     psnr_gt.append(gt[start, :])
    #                     psnr_prediction.append(prediction[start, :])
    # else:
    #
    #     if os.path.exists(args.brain_max_lenght):
    #         list_lenght = np.load(args.brain_max_lenght)
    #         xx_min, xx_max, yy_min, yy_max = list_lenght[0], list_lenght[1], list_lenght[2], list_lenght[3]
    #     else:
    #         print('Missing brain length file')
    #
    #     edge_distance_x_start = xx_min
    #     edge_distance_x_ed = xx_max
    #     x_length = xx_max - xx_min
    #     x_final_length = math.ceil(x_length / 10) * 10
    #     x_add_length = x_final_length - x_length
    #     x_left_add_length = math.floor(x_add_length / 2)
    #
    #     edge_distance_y_start = yy_min
    #     edge_distance_y_ed = yy_max
    #     y_length = yy_max - yy_min
    #     y_final_length = math.ceil(y_length / 10) * 10
    #     y_add_length = y_final_length - y_length
    #     y_left_add_length = math.floor(y_add_length / 2)
    #
    #     for i in range(0, z_size, 1):
    #         prediction_like[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i, :] = prediction[i, x_left_add_length:x_left_add_length + x_length, y_left_add_length:y_left_add_length + y_length, :]
    #         gt_like[edge_distance_x_start:edge_distance_x_ed, edge_distance_y_start:edge_distance_y_ed, i, :] = gt[i, x_left_add_length:x_left_add_length + x_length, y_left_add_length:y_left_add_length + y_length, :]
    #
    #
    #     for xx in range(edge_distance_x_start, edge_distance_x_ed, 1):
    #         for yy in range(edge_distance_y_start, edge_distance_y_ed, 1):
    #             for zz in range(0, z_size, 1):
    #                 if mask[xx, yy, zz] > 0:
    #                     psnr_gt.append(gt_like[xx, yy, zz, :])
    #                     psnr_prediction.append(prediction_like[xx, yy, zz, :])
    #                 else:
    #                     gt_like[xx, yy, zz, :] = 0
    #                     prediction_like[xx, yy, zz, :] = 0

    psnr_gt = np.array(gt)
    psnr_prediction = np.array(prediction)

    # if args.is_generate_image:
    #     args.generate_image_save_path = os.path.join(args.generate_image_save_path, subject_id + str(time.time()) + 'nii.gz' )
    #     affine = np.eye(4)
    #     img_prediction = nib.Nifti1Image(prediction_like, affine)
    #     nib.save(img_prediction, args.generate_image_save_path)
    #     print('save img successful')
    return computer_psnr(psnr_gt, psnr_prediction, args.microstructure_name, args.is_train)