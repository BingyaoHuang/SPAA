'''
PCNet and CompenNet++ training functions
'''

import os
from os.path import join
import warnings
import numpy as np
import cv2 as cv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pytorch_ssim
import models
from img_proc import threshold_im, center_crop as cc
from utils import vis, plot_montage, append_data_point
import utils as ut
from omegaconf import OmegaConf, DictConfig

# l1_fun = nn.L1Loss()
# l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda() if torch.cuda.is_available() else pytorch_ssim.SSIM()


def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def load_data(data_root, setup_name, input_size=None, compensation=False):
    # data paths
    setup_path     = join(data_root , 'setups', setup_name)
    cam_ref_path   = join(setup_path, 'cam/raw/ref')
    cam_train_path = join(setup_path, 'cam/raw/train')
    cam_valid_path = join(setup_path, 'cam/raw/test')
    cam_cb_path    = join(setup_path, 'cam/raw/cb')
    prj_train_path = join(data_root , 'prj_share/train')
    prj_valid_path = join(data_root , 'prj_share/test')
    print("Loading data from '{}'".format(setup_path))

    # load setup_info
    setup_info = load_setup_info(setup_path)

    # ref
    cam_ref = ut.torch_imread_mt(cam_ref_path, size=input_size)

    # ref image with gray illumination
    gray_idx = 1  # cam/raw/ref/img_0002, lit by prj_brightness, cam-captured surface i.e., s when img_gray projected
    cam_scene = cam_ref[gray_idx].unsqueeze(0)  # ref/img_0002 is

    # textured train
    cam_train = ut.torch_imread_mt(cam_train_path, size=input_size)
    prj_train = ut.torch_imread_mt(prj_train_path)

    # test/validation data
    cam_valid = ut.torch_imread_mt(cam_valid_path, size=input_size)
    prj_valid = ut.torch_imread_mt(prj_valid_path, index=[i for i in range(cam_valid.shape[0])])  # only read the same number of cam_valid

    # find projector direct light mask
    im_cb = ut.torch_imread_mt(cam_cb_path, size=input_size)
    im_cb = im_cb.numpy().transpose((2, 3, 1, 0))

    # find direct light mask using Nayar's TOG'06 method (also see Moreno 3DV'12)
    l1 = im_cb.max(axis=3)  # max image L+
    l2 = im_cb.min(axis=3)  # max image L-
    b = 0.9  # projector backlight strength (for mask use a large b, for real direct/indirect separation, use a smaller b)
    im_direct   = (l1 - l2) / (1 - b)  # direct light image
    im_indirect = 2 * (l2 - b * l1) / (1 - b * b)  # indirect (global) light image

    im_mask, _, mask_corners = threshold_im(im_direct, compensation=compensation)  # use thresholded as mask
    im_mask = torch.Tensor(im_mask).bool()

    return cam_scene, cam_train, cam_valid, prj_train, prj_valid, im_mask, mask_corners, setup_info


def load_setup_info(setup_path):
    # setup_info
    setup_info_filename = join(setup_path, 'setup_info.yml')
    if os.path.exists(setup_info_filename):
        setup_info = OmegaConf.load(setup_info_filename)
        print(f'{setup_info_filename} loaded')
    else:
        setup_info_default_filename = join(setup_path, '../setup_info_default.yml')
        setup_info = OmegaConf.load(setup_info_default_filename)
        warnings.warn(f'{setup_info_filename} not found, loading {setup_info_default_filename} instead')
    return setup_info


def init_compennet(compennet, data_root, cfg):
    # initialize CompenNet to |x-s| without actual projections
    ckpt_file = join(data_root, '../checkpoint/init_CompenNet_l1+ssim_500_48_500_0.001_0.2_800_0.0001.pth')

    if os.path.exists(ckpt_file):
        # load weights initialized CompenNet from saved state dict
        compennet.load_state_dict(torch.load(ckpt_file))

        print('CompenNet state dict found! Loading...')
    else:
        # initialize the model if checkpoint file does not exist
        print('CompenNet state dict not found! Initializing...')
        cam_scene_init_path = join(data_root, 'prj_share/init')
        prj_train_path      = join(data_root, 'prj_share/train')

        # load data
        cam_scene = ut.torch_imread_mt(cam_scene_init_path)
        prj_train = ut.torch_imread_mt(prj_train_path)
        init_data = dict(cam_scene=cam_scene,
                         cam_train=torch.abs(prj_train - 0.3 * cam_scene.expand_as(prj_train)),
                         prj_train=prj_train)

        # then initialize compenNet to |x-s|
        init_cfg = DictConfig({'data_root': data_root, 'setup_name': 'init', 'num_dataset': 1, 'device': cfg.device, 'max_epochs': 2000,
                               'max_iters': 500, 'batch_size': 48, 'lr': 1e-3, 'lr_drop_ratio': 0.2, 'lr_drop_rate': 800, 'loss': 'l1+ssim',
                               'l2_reg'   : 1e-4, 'plot_on': True, 'train_plot_rate': 50, 'valid_rate': 200, 'num_train': 500})

        compennet, _, _, _ = train_compennet_pp(compennet, init_data, None, init_cfg)

    return compennet


def train_compennet_pp(model, train_data, valid_data, cfg):
    device = cfg['device']

    # training data
    cam_scene_train = train_data['cam_scene'].to(device)
    cam_train       = train_data['cam_train']
    prj_train       = train_data['prj_train']

    # only use one surf
    cam_scene_train_batch = cam_scene_train.expand(cfg.batch_size, -1, -1, -1)

    # list of parameters to be optimized
    params = filter(lambda param: param.requires_grad, model.parameters())  # only optimize parameters that require gradient

    # optimizer
    optimizer = optim.Adam(params, lr=cfg['lr'], weight_decay=cfg['l2_reg'])

    # learning rate drop scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_drop_rate'], gamma=cfg['lr_drop_ratio'])

    # %% start train
    start_time = time.time()

    # get model name
    if 'model_name' not in cfg: cfg['model_name'] = model.name if hasattr(model, 'name') else model.module.name

    # initialize visdom data visualization figure
    if 'plot_on' not in cfg: cfg['plot_on'] = True

    # title string of current training option
    title = ut.opt_to_string(cfg)

    if cfg['plot_on']:
        # intialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                 opts=dict(width=1300, height=500, markers=True, markersize=3,
                                           layoutopts=dict(
                                               plotly=dict(title={'text': title, 'font': {'size': 24}},
                                                           font={'family': 'Arial', 'size': 20},
                                                           hoverlabel={'font': {'size': 20}}, hovermode='x',
                                                           xaxis={'title': 'Iteration'},
                                                           yaxis={'title': 'Metrics', 'hoverformat': '.4f'}))))
    # main loop
    iters = 0

    while iters < cfg['max_iters']:
        # randomly sample training batch and send to GPU
        idx = random.sample(range(cfg.num_train), cfg.batch_size)
        cam_train_batch = cam_train[idx].to(device) if cam_train.device.type != 'cuda' else cam_train[idx]
        prj_train_batch = prj_train[idx].to(device) if prj_train.device.type != 'cuda' else prj_train[idx]

        # infer and compute loss
        model.train()  # explicitly set to train mode in case batchNormalization and dropout are used
        prj_train_infer = model(cam_train_batch, cam_scene_train_batch)
        train_loss_batch, train_l2_loss_batch = compute_loss(prj_train_infer, prj_train_batch, cfg['loss'])
        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channel, rgb

        # backpropagation and update params
        optimizer.zero_grad()
        train_loss_batch.backward()
        optimizer.step()

        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        # plot train
        if cfg['plot_on']:
            vis_idx = range(5)
            # cp_sz = cfg.setup_info.classifier_crop_sz  # visualize center crop
            if iters % cfg['train_plot_rate'] == 0 or iters == cfg['max_iters'] - 1:
                vis_train_fig = plot_montage(cam_train_batch[vis_idx], prj_train_infer[vis_idx], prj_train_batch[vis_idx], win=vis_train_fig,
                                             title='[Train]' + title)
                append_data_point(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                append_data_point(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')

        # validation
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        if valid_data is not None and (iters % cfg['valid_rate'] == 0 or iters == cfg['max_iters'] - 1):
            valid_psnr, valid_rmse, valid_ssim, prj_valid_infer = evaluate_model(model, valid_data)

            # plot validation
            if cfg['plot_on']:
                idx = np.array([9, 10, 11, 14, 70]) - 1
                vis_valid_fig = plot_montage(valid_data['cam_valid'][idx], prj_valid_infer[idx], valid_data['prj_valid'][idx], win=vis_valid_fig,
                                             title='[Valid]' + title)
                append_data_point(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                append_data_point(iters, valid_ssim, vis_curve_fig, 'valid_ssim')

        # print to console
        print(
            f"Iter:{iters:5d} | Time: {time_lapse} | Train Loss: {train_loss_batch.item():.4f} | Train RMSE: {train_rmse_batch:.4f} "
            f"| Valid PSNR: {f'{valid_psnr:>2.4f}' if valid_psnr else '':7s}  | Valid RMSE: {f'{valid_rmse:.4f}' if valid_rmse else '':6s}  "
            f"| Valid SSIM: {f'{valid_ssim:.4f}' if valid_ssim else '':6s}  | Learn Rate: {optimizer.param_groups[0]['lr']:.5f} |")

        lr_scheduler.step()  # update learning rate according to schedule
        iters += 1

    # Done training and save the last epoch model
    ut.save_checkpoint(join(cfg.data_root, '../checkpoint'), model, title)

    return model, valid_psnr, valid_rmse, valid_ssim


def train_pcnet(model, train_data, valid_data, cfg):
    device = torch.device(cfg.device)

    # training data
    # cam_mask        = train_data['mask'].to(device) # thresholded camera-captured surface image, s*
    cam_scene_train = train_data['cam_scene'].to(device)
    cam_train       = train_data['cam_train']
    prj_train       = train_data['prj_train']

    # only use one surf
    cam_scene_train_batch = cam_scene_train.expand(cfg.batch_size, -1, -1, -1)

    # params, optimizers and lr schedulers
    aff_tps_params    = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ['module.warping_net.affine_mat', 'module.warping_net.theta'] , model.named_parameters()))))
    refinenet_params  = list(map(lambda x: x[1], list(filter(lambda kv: 'module.warping_net.grid_refine_net' in kv[0], model.named_parameters()))))
    shadingnet_params = list(map(lambda x: x[1], list(filter(lambda kv: 'module.warping_net' not in kv[0]            , model.named_parameters()))))

    if 1:
        w1_optimizer = optim.Adam([{'params': aff_tps_params}]   , lr = 1e-2, weight_decay = 0)
        w2_optimizer = optim.Adam([{'params': refinenet_params}] , lr = 5e-3, weight_decay = 0)
        s_optimizer  = optim.Adam([{'params': shadingnet_params}], lr = 1e-3, weight_decay = cfg.l2_reg)
    else:
        # for pretrained PCNet finetune
        w1_optimizer = optim.Adam([{'params': aff_tps_params}]   , lr = 1e-3, weight_decay = 0)
        w2_optimizer = optim.Adam([{'params': refinenet_params}] , lr = 1e-2, weight_decay = 0)
        s_optimizer  = optim.Adam([{'params': shadingnet_params}], lr = 1e-4, weight_decay = cfg.l2_reg)

    # learning rate drop scheduler
    w1_lr_scheduler = optim.lr_scheduler.MultiStepLR(w1_optimizer, milestones = [100],  gamma = cfg.lr_drop_ratio)
    w2_lr_scheduler = optim.lr_scheduler.MultiStepLR(w2_optimizer, milestones = [1200], gamma = cfg.lr_drop_ratio)
    s_lr_scheduler  = optim.lr_scheduler.MultiStepLR(s_optimizer , milestones = [1800], gamma = cfg.lr_drop_ratio)

    # %% start train
    start_time = time.time()

    # get model name
    if 'model_name' not in cfg: cfg.model_name = model.name if hasattr(model, 'name') else model.module.name

    # initialize visdom data visualization figure
    if 'plot_on' not in cfg: cfg.plot_on = True

    # title string of current training option
    title = ut.opt_to_string(cfg)

    # initialize visdom figures
    if cfg.plot_on:
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                 opts=dict(width=1300, height=500, markers=True, markersize=3,
                                           layoutopts=dict(
                                               plotly=dict(title={'text': title, 'font': {'size': 24}},
                                                           font={'family': 'Arial', 'size': 20},
                                                           hoverlabel={'font': {'size': 20}}, hovermode='x',
                                                           xaxis={'title': 'Iteration'},
                                                           yaxis={'title': 'Metrics', 'hoverformat': '.4f'}))))
    # main loop
    iters = 0
    while iters < cfg.max_iters:
        # randomly sample training batch and send to GPU
        idx = random.sample(range(cfg.num_train), cfg.batch_size)
        cam_train_batch = cam_train[idx].to(device) if cam_train.device.type != 'cuda' else cam_train[idx]
        prj_train_batch = prj_train[idx].to(device) if prj_train.device.type != 'cuda' else prj_train[idx]

        # good for faster convergence and avoid early local minima due to ssim loss
        if iters <= 400:
            cfg.loss = 'l1'
        else:
            cfg.loss = 'l1+ssim'

        # infer and compute loss
        model.train()  # explicitly set to train mode in case batchNormalization and dropout are used
        cam_train_infer = model(prj_train_batch, cam_scene_train_batch)
        train_loss_batch, train_l2_loss_batch = compute_loss(cam_train_infer, cam_train_batch, cfg.loss)
        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channel, rgb

        # backpropagation and update params
        w1_optimizer.zero_grad()
        w2_optimizer.zero_grad()
        s_optimizer.zero_grad()

        train_loss_batch.backward()

        w1_optimizer.step()
        w2_optimizer.step()
        s_optimizer.step()

        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        # plot train
        if cfg.plot_on:
            vis_idx = range(5)
            # cp_sz = cfg.setup_info.classifier_crop_sz  # visualize center crop
            if iters % cfg.train_plot_rate == 0 or iters == cfg.max_iters - 1:
                vis_train_fig = plot_montage(prj_train_batch[vis_idx], cam_train_infer[vis_idx], cam_train_batch[vis_idx], win=vis_train_fig,
                                             title='[Train]' + title)
                append_data_point(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                append_data_point(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')

        # validation
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        if valid_data is not None and (iters % cfg.valid_rate == 0 or iters == cfg.max_iters - 1):
            # valid_psnr, valid_rmse, valid_ssim, cam_valid_infer = evaluate_pcnet(model, valid_data)
            valid_psnr, valid_rmse, valid_ssim, cam_valid_infer = evaluate_model(model, valid_data)

            # plot validation
            if cfg.plot_on:
                vis_idx = np.array([9, 10, 11, 14, 70]) - 1
                vis_valid_fig = plot_montage(valid_data['prj_valid'][vis_idx], cam_valid_infer[vis_idx], valid_data['cam_valid'][vis_idx],
                                             win=vis_valid_fig, title='[Valid]' + title)
                append_data_point(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                append_data_point(iters, valid_ssim, vis_curve_fig, 'valid_ssim')

        print(f"Iter:{iters:5d} | Time: {time_lapse} | Train Loss: {train_loss_batch.item():.4f} | Train RMSE: {train_rmse_batch:.4f} "
            f"| Valid PSNR: {f'{valid_psnr:>2.4f}' if valid_psnr else '':7s}  | Valid RMSE: {f'{valid_rmse:.4f}' if valid_rmse else '':6s} "
            f"| Valid SSIM: {f'{valid_ssim:.4f}' if valid_ssim else '':6s}  "
            f"| Learn Rate: {w2_optimizer.param_groups[0]['lr']:.5f}/{s_optimizer.param_groups[0]['lr']:.5f} |")

        # update learning rates according to schedule
        w1_lr_scheduler.step()
        w2_lr_scheduler.step()
        s_lr_scheduler.step()
        iters += 1

    # Done training and save the last epoch model
    ut.save_checkpoint(join(cfg.data_root, '../checkpoint'), model, title)

    return model, valid_psnr, valid_rmse, valid_ssim


# compute loss between inference and ground truth
def compute_loss(prj_infer, prj_train, loss_option):
    if loss_option == '':
        raise TypeError('Loss type not specified')

    train_loss = 0

    # l1
    if 'l1' in loss_option:
        l1_loss = F.l1_loss(prj_infer, prj_train, reduction='mean')
        train_loss += l1_loss

    # l2
    l2_loss = F.mse_loss(prj_infer, prj_train, reduction='mean')
    if 'l2' in loss_option:
        train_loss += l2_loss

    # ssim
    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_infer, prj_train))
        train_loss += ssim_loss

    if 'huber' in loss_option:
        huber_loss = huber(prj_infer, prj_train).abs().mean()
        train_loss += huber_loss

    return train_loss, l2_loss


def evaluate_model(model, valid_data, chunk_sz=10):
    # evaluate model on validation dataset
    # increase chunk_sz when you encounter out of GPU memory issue
    cam_scene = valid_data['cam_scene']
    cam_valid = valid_data['cam_valid']
    prj_valid = valid_data['prj_valid']

    device = cam_valid.device

    with torch.no_grad():
        model.eval()  # explicitly set to eval mode

        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.

        if 'PCNet' in model.module.name:
            model_infer = torch.zeros(cam_valid.shape)
        elif 'CompenNet++' in model.module.name:
            model_infer = torch.zeros(prj_valid.shape)

        num_valid = cam_valid.shape[0]
        batch_idx = torch.chunk(torch.arange(num_valid), chunk_sz)  # chunks = num_valid sets batch size to 1

        for idx in batch_idx:
            batch_size = len(idx)
            cam_scene_batch = cam_scene[idx].to(device) if cam_scene.device.type != 'cuda' else cam_scene[idx]
            cam_valid_batch = cam_valid[idx].to(device) if cam_valid.device.type != 'cuda' else cam_valid[idx]
            prj_valid_batch = prj_valid[idx].to(device) if prj_valid.device.type != 'cuda' else prj_valid[idx]

            if 'PCNet' in model.module.name:
                model_input_batch = prj_valid_batch
                valid_gt_batch    = cam_valid_batch
            elif 'CompenNet++' in model.module.name:
                model_input_batch = cam_valid_batch
                valid_gt_batch    = prj_valid_batch

            # infer batch
            model_infer_batch = model(model_input_batch, cam_scene_batch)
            if type(model_infer_batch) == tuple and len(model_infer_batch) > 1: model_infer_batch = model_infer_batch[0]
            model_infer[idx] = model_infer_batch.detach().cpu()

            # compute metrics
            valid_metrics_batch  = ut.calc_img_dists(model_infer_batch, valid_gt_batch)
            valid_psnr += valid_metrics_batch[0] * batch_size / num_valid
            valid_rmse += valid_metrics_batch[1] * batch_size / num_valid
            valid_ssim += valid_metrics_batch[2] * batch_size / num_valid

    return valid_psnr, valid_rmse, valid_ssim, model_infer


def get_model_train_cfg(model_list, data_root=None, setup_list=None, device_ids=[0], center_crop=False, load_pretrained=False, plot_on=True,
                        single=False):
    # default training configs
    cfg_default                    = DictConfig({})
    cfg_default.data_root          = data_root
    cfg_default.setup_list         = setup_list
    cfg_default.device             = 'cuda'
    cfg_default.device_ids         = device_ids
    cfg_default.load_pretrained    = load_pretrained
    cfg_default.max_iters          = 2000
    cfg_default.batch_size         = 24
    cfg_default.lr                 = 1e-3
    cfg_default.lr_drop_ratio      = 0.2
    cfg_default.lr_drop_rate       = 800            # TODO PCNet uses its own lr_drop_rate, and this is ignored
    cfg_default.l2_reg             = 1e-4
    cfg_default.train_plot_rate    = 50
    cfg_default.valid_rate         = 200            # validation and visdom plot rate (use a larger valid_rate to save running time)
    cfg_default.plot_on            = plot_on        # disable when running stats for all setups
    cfg_default.center_crop        = center_crop    # whether to center crop the training/validation images

    if single:                     # single model, no list
        cfg_default.model_name     = model_list[0]
        cfg_default.num_train      = 500
        cfg_default.loss           = 'l1+ssim'
    else:
        cfg_default.model_list     = model_list
        cfg_default.num_train_list = [500]
        cfg_default.loss_list      = ['l1+ssim']

    return cfg_default


def train_eval_pcnet(cfg_default):
    data_root = cfg_default.data_root

    # set PyTorch device to GPU
    device = torch.device(cfg_default.device)

    # log
    ret, log_txt_filename, log_xls_filename = ut.init_log_file(join(data_root, '../log'))

    # train and evaluate all setups
    for setup_name in cfg_default.setup_list:
        # load training and validation data
        cam_scene, cam_train, cam_valid, prj_train, prj_valid, cam_mask, mask_corners, setup_info = load_data(data_root, setup_name)
        cfg_default.setup_info = setup_info

        # center crop, decide whether PCNet output is center cropped square image (classifier_crop_sz) or not (cam_im_sz)
        if cfg_default.center_crop:
            cp_sz = setup_info.classifier_crop_sz
            cam_scene = cc(cam_scene, cp_sz)
            cam_train = cc(cam_train, cp_sz)
            cam_valid = cc(cam_valid, cp_sz)
            cam_mask  = cc(cam_mask , cp_sz)

        # surface image for training and validation
        cam_scene = cam_scene.to(device)
        cam_scene_train = cam_scene
        cam_scene_valid = cam_scene.expand(cam_valid.shape[0], -1, -1, -1)

        # convert valid data to CUDA tensor if you have sufficient GPU memory (significant speedup)
        cam_valid = cam_valid.to(device)
        prj_valid = prj_valid.to(device)

        # validation data, 200 image pairs
        valid_data = dict(cam_scene=cam_scene_valid, cam_valid=cam_valid, prj_valid=prj_valid)

        # stats for different #Train
        for num_train in cfg_default.num_train_list:
            cfg = cfg_default.copy()
            cfg.num_train = num_train
            for key in ['num_train_list', 'model_list', 'loss_list', 'setup_list']:
                del cfg[key]

            # select a subset to train
            train_data = dict(cam_scene=cam_scene_train, cam_train=cam_train[:num_train], prj_train=prj_train[:num_train], mask=cam_mask)

            # stats for different models
            for model_name in cfg_default.model_list:
                cfg.model_name = model_name.replace('/', '_')

                # stats for different loss functions
                for loss in cfg_default.loss_list:
                    # train option for current configuration, i.e., setup name and loss function
                    cfg.setup_name = setup_name.replace('/', '_')
                    cfg.loss = loss
                    model_version = f'{cfg.model_name}_{loss}_{num_train}_{cfg.batch_size}_{cfg.max_iters}'

                    # set seed of rng for repeatability
                    ut.reset_rng_seeds(123)

                    # create a ShadingNetSPAA model
                    shading_net = models.ShadingNetSPAA(use_rough='no_rough' not in model_name)
                    if torch.cuda.device_count() >= 1: shading_net = nn.DataParallel(shading_net, device_ids=cfg.device_ids).to(device)

                    # create a WarpingNet model
                    warping_net = models.WarpingNet(out_size=cam_train.shape[-2:], with_refine='w/o_refine' not in model_name)  # warp prj to cam raw

                    # initialize WarpingNet with affine transformation (remember grid_sample is inverse warp, so src is the desired warp
                    src_pts    = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                    dst_pts    = np.array(mask_corners[0:3]).astype(np.float32)
                    affine_mat = torch.Tensor(cv.getAffineTransform(dst_pts, src_pts))  # prj -> cam
                    warping_net.set_affine(affine_mat.flatten())
                    if torch.cuda.device_count() >= 1: warping_net = nn.DataParallel(warping_net, device_ids=cfg.device_ids).to(device)

                    # create a PCNet model using WarpingNet and ShadingNetSPAA
                    pcnet = models.PCNet(cam_mask.float(), warping_net, shading_net, fix_shading_net=False, use_mask='no_mask' not in model_name,
                                         use_rough='no_rough' not in model_name)
                    if torch.cuda.device_count() >= 1: pcnet = nn.DataParallel(pcnet, device_ids=cfg.device_ids).to(device)

                    print('-------------------------------------- Training Options -----------------------------------')
                    print('\n'.join(f'{k}: {v}' for k, v in cfg.items()))

                    if not cfg.load_pretrained:
                        print(f'------------------------------------ Start training {model_name:s} ---------------------------')
                        pcnet, valid_psnr, valid_rmse, valid_ssim = train_pcnet(pcnet, train_data, valid_data, cfg)
                    else:
                        # load the previously trained PCNet instead of retraining a new one
                        print(f'------------------------------------ Loading pretrained {model_name:s} ---------------------------')
                        checkpoint_filename = join(data_root, '../checkpoint', ut.opt_to_string(cfg) + '.pth')
                        pcnet.load_state_dict(torch.load(checkpoint_filename))

                    # [validation phase] after training we evaluate and save results
                    # cam_valid_infer = evaluate_pcnet(pcnet, valid_data)[-1]
                    cam_valid_infer = evaluate_model(pcnet, valid_data)[-1]
                    valid_psnr, valid_rmse, valid_ssim, valid_l2, valid_linf, valid_dE = ut.calc_img_dists(cam_valid_infer, cam_valid)

                    # save results to log file
                    ret.loc[len(ret)] = [setup_name, model_name, loss, num_train, cfg.batch_size, cfg.max_iters, valid_psnr, valid_rmse,
                                         valid_ssim, valid_l2, valid_linf, valid_dE]
                    ut.write_log_file(ret, log_txt_filename, log_xls_filename)  # in case unexpected interruptions, we save logs for each setup

                    # save inferred camera-captured (relit) images
                    setup_path = join(data_root, 'setups', setup_name)
                    cam_valid_infer_path = join(setup_path, 'cam/infer/test', model_version)
                    ut.save_imgs(cam_valid_infer.detach().cpu(), cam_valid_infer_path)
                    print('Inferred camera-captured (relit) images saved to ' + cam_valid_infer_path)
                    print('------------------------------------ Done! ---------------------------\n')

    # average all setups' metrics and save to log
    for model_name in cfg_default.model_list:
        ret.loc[len(ret)] = ret.loc[ret['Model'] == model_name].mean(axis=0, numeric_only=True)
        ret.loc[len(ret) - 1, ['Setup', 'Model']] = [f'[mean]_{len(cfg_default.setup_list)}_setups', model_name]

    # ret.loc[len(ret)] = ret.mean(axis=0, numeric_only=True)
    # ret.loc[len(ret) - 1, 'Setup'] = '[mean]'
    print(ret.to_string(justify='center', float_format='%.4f'))
    print('-------------------------------------- End of result table ---------------------------\n')
    ut.write_log_file(ret, log_txt_filename, log_xls_filename)  # log of all setups

    return pcnet, ret, cfg


def train_eval_compennet_pp(cfg_default):
    data_root = cfg_default.data_root

    # set PyTorch device to GPU
    device = torch.device(cfg_default.device)

    # log
    ret, log_txt_filename, log_xls_filename = ut.init_log_file(join(data_root, '../log'))

    # initialize CompenNet by loading the weights if it exists, otherwise quickly pretrain CompenNet
    compen_net = models.CompenNet()
    if torch.cuda.device_count() >= 1: compen_net = nn.DataParallel(compen_net, device_ids=cfg_default.device_ids).to(device)
    compen_net = init_compennet(compen_net, data_root, cfg_default)

    # train and evaluate all setups
    for setup_name in cfg_default.setup_list:
        # load training and validation data
        setup_path = join(data_root, 'setups', setup_name)
        cam_scene, cam_train, cam_valid, prj_train, prj_valid, cam_mask, mask_corners, setup_info = load_data(data_root, setup_name)
        cfg_default.setup_info = setup_info

        prj_size = prj_train.shape[2:4]

        # surface image for training and validation
        cam_scene = cam_scene.to(device)
        cam_scene_train = cam_scene
        cam_scene_valid = cam_scene.expand(cam_valid.shape[0], -1, -1, -1)

        # convert valid data to CUDA tensor if you have sufficient GPU memory (significant speedup)
        cam_valid = cam_valid.to(device)
        prj_valid = prj_valid.to(device)

        # validation data, 200 image pairs
        valid_data = dict(cam_scene=cam_scene_valid, cam_valid=cam_valid, prj_valid=prj_valid)

        # stats for different #Train
        for num_train in cfg_default.num_train_list:
            cfg = cfg_default.copy()
            cfg.num_train = num_train
            for key in ['num_train_list', 'model_list', 'loss_list', 'setup_list']:
                del cfg[key]

            # select a subset to train
            train_data = dict(cam_scene=cam_scene_train, cam_train=cam_train[:num_train], prj_train=prj_train[:num_train], mask=cam_mask)

            # stats for different models
            for model_name in cfg_default.model_list:
                cfg.model_name = model_name.replace('/', '_')

                # stats for different loss functions
                for loss in cfg_default.loss_list:
                    # train option for current configuration, i.e., setup name and loss function
                    cfg.setup_name = setup_name.replace('/', '_')
                    cfg.loss = loss
                    model_version = f'{cfg.model_name}_{loss}_{num_train}_{cfg.batch_size}_{cfg.max_iters}'

                    # set seed of rng for repeatability
                    ut.reset_rng_seeds(0)

                    # create a WarpingNet model
                    warping_net = models.WarpingNet(out_size=prj_size, with_refine='w/o_refine' not in model_name)  # warp prj to cam raw

                    # initialize WarpingNet with affine transformation (remember grid_sample is inverse warp, so src is the desired warp
                    src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
                    dst_pts = np.array(mask_corners[0:3]).astype(np.float32)
                    affine_mat = torch.Tensor(cv.getAffineTransform(dst_pts, src_pts))  # prj -> cam
                    warping_net.set_affine(affine_mat.flatten())
                    if torch.cuda.device_count() >= 1: warping_net = nn.DataParallel(warping_net, device_ids=cfg.device_ids).to(device)

                    # create a CompenNet++ model from existing WarpingNet and CompenNet
                    compennet_pp = models.CompenNetPlusplus(warping_net, compen_net)
                    if torch.cuda.device_count() >= 1: compennet_pp = nn.DataParallel(compennet_pp, device_ids=cfg.device_ids).to(device)

                    print('-------------------------------------- Training Options -----------------------------------')
                    print('\n'.join(f'{k}: {v}' for k, v in cfg.items()))

                    if not cfg.load_pretrained:
                        print(f'------------------------------------ Start training {model_name:s} ---------------------------')
                        compennet_pp, valid_psnr, valid_rmse, valid_ssim = train_compennet_pp(compennet_pp, train_data, valid_data, cfg)
                    else:
                        # load the previously trained PCNet instead of retraining a new one
                        print(f'------------------------------------ Loading pretrained {model_name:s} ---------------------------')
                        checkpoint_filename = join(data_root, '../checkpoint', ut.opt_to_string(cfg) + '.pth')
                        compennet_pp.load_state_dict(torch.load(checkpoint_filename))

                    # [validation phase] after training we evaluate and save results
                    prj_valid_infer = evaluate_model(compennet_pp, valid_data)[-1]
                    valid_psnr, valid_rmse, valid_ssim, valid_l2, valid_linf, valid_dE = ut.calc_img_dists(prj_valid_infer, prj_valid)

                    # save results to log file
                    ret.loc[len(ret)] = [setup_name, model_name, loss, num_train, cfg.batch_size, cfg.max_iters, valid_psnr, valid_rmse,
                                         valid_ssim, valid_l2, valid_linf, valid_dE]
                    ut.write_log_file(ret, log_txt_filename, log_xls_filename)

                    # save inferred projector input images
                    prj_valid_infer_path = join(setup_path, 'prj/infer/test', model_version)
                    ut.save_imgs(prj_valid_infer, prj_valid_infer_path)
                    print('Inferred projector input validation images saved to ' + prj_valid_infer_path)

                    # [testing phase] create compensated test images for real compensation (to project and capture)
                    print(f'------------------------------------ Saving compensated test images for {model_name:s} ---------------------------')

                    # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
                    desire_test_path = join(setup_path, 'cam/desire/test')
                    if os.path.isdir(desire_test_path):
                        # compensate and save images
                        desire_test    = ut.torch_imread_mt(desire_test_path).to(device)
                        cam_scene_test = cam_scene.expand_as(desire_test).to(device)
                        with torch.no_grad():
                            # simplify CompenNet++
                            compennet_pp.module.simplify(cam_scene_test[0, ...].unsqueeze(0))

                            # compensate using CompenNet++
                            compennet_pp.eval()
                            prj_cmp_test = compennet_pp(desire_test, cam_scene_test).detach()  # compensated prj input image x^{*}
                        del desire_test, cam_scene_test

                        # save images
                        prj_cmp_path = join(setup_path, 'prj/cmp/test', model_version)
                        ut.save_imgs(prj_cmp_test, prj_cmp_path)  # compensated testing images, i.e., to be projected to the surface
                        print('Compensation images saved to ' + prj_cmp_path)
                    else:
                        warnings.warn(f'images and folder {desire_test_path:s} does not exist, no compensation images saved!')
                    print('------------------------------------ Done! ---------------------------\n')

    # average all setups' metrics and save to log
    for model_name in cfg_default.model_list:
        ret.loc[len(ret)] = ret.loc[ret['Model'] == model_name].mean(axis=0, numeric_only=True)
        ret.loc[len(ret) - 1, ['Setup', 'Model']] = [f'[mean]_{len(cfg_default.setup_list)}_setups', model_name]

    # ret.loc[len(ret)] = ret.mean(axis=0, numeric_only=True)
    # ret.loc[len(ret) - 1, 'Setup'] = '[mean]'
    print(ret.to_string(justify='center', float_format='%.4f'))
    print('-------------------------------------- End of result table ---------------------------\n')
    ut.write_log_file(ret, log_txt_filename, log_xls_filename)

    return compennet_pp, ret, cfg
