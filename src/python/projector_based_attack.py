"""
Useful functions for projector-based adversarial attack
"""
import os
from os.path import join
import numpy as np
import cv2 as cv
import pandas as pd
import torch
from omegaconf import DictConfig
from torchvision.utils import make_grid
from train_network import load_setup_info, train_eval_pcnet, train_eval_compennet_pp, get_model_train_cfg
from img_proc import resize, insert_text, expand_boarder, expand_4d, center_crop as cc
import utils as ut
from utils import calc_img_dists
from perc_al.differential_color_functions import rgb2lab_diff, ciede2000_diff
from perc_al import PerC_AL
import itertools
from classifier import Classifier, load_imagenet_labels
from one_pixel_attacker import ProjectorOnePixelAttacker
from tqdm import tqdm


def run_projector_based_attack(cfg):
    attacker_name = cfg.attacker_name
    assert attacker_name in ['SPAA', 'PerC-AL+CompenNet++', 'One-pixel_DE'], f'{attacker_name} not supported!'
    assert (attacker_name != 'One-pixel_DE') or (len(cfg.setup_list) == 1), f'{attacker_name} does not support attacking multiple setups simultaneously!'

    # set PyTorch device to GPU
    device = torch.device(cfg.device)
    ut.reset_rng_seeds(0)

    for setup_name in cfg.setup_list:
        print(f'\nPerforming [{attacker_name}] attack on [{setup_name}]')

        # load setup info and images
        setup_path = join(cfg.data_root, 'setups', setup_name)
        setup_info = load_setup_info(setup_path)
        cp_sz = setup_info.classifier_crop_sz
        cam_scene  = cc(ut.torch_imread(join(setup_path, 'cam/raw/ref/img_0002.png')), setup_info.cam_im_sz[::-1]) # cam-captured scene (Is), ref/img_0002

        # ImageNet and targeted attack labels
        imagenet_labels = load_imagenet_labels(join(cfg.data_root, 'imagenet1000_clsidx_to_labels.txt'))
        target_labels   = load_imagenet_labels(join(cfg.data_root, 'imagenet10_clsidx_to_labels.txt'))  # targeted attack labels

        dl_based = attacker_name in ['SPAA', 'PerC-AL+CompenNet++']
        if dl_based:
            cam_scene = cam_scene.to(device)

            # train or load PCNet/CompenNet++ model
            model_cfg = get_model_train_cfg(model_list=None, data_root=cfg.data_root, setup_list=[setup_name], device_ids=cfg.device_ids,
                                            load_pretrained=cfg.load_pretrained, plot_on=cfg.plot_on)
            if attacker_name == 'SPAA':
                model_cfg.model_list = ['PCNet']
                # model_cfg.max_iters = 100 # debug
                model, model_ret, model_cfg = train_eval_pcnet(model_cfg)
            elif attacker_name == 'PerC-AL+CompenNet++':
                model_cfg.model_list = ['CompenNet++']
                # model_cfg.max_iters = 100 # debug
                model, model_ret, model_cfg = train_eval_compennet_pp(model_cfg)

            # set to evaluation mode
            model.eval()

            # fix model weights
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Nichols & Jasper's projector-based One-pixel DE attacker
            one_pixel_de = ProjectorOnePixelAttacker(imagenet_labels, setup_info)
            im_prj_org = setup_info['prj_brightness'] * torch.ones(3, *setup_info['prj_im_sz'])
            one_pixel_de.im_cam_org = cam_scene
            model_cfg = None  # no deep learning-based models

        attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
        cfg.model_cfg = model_cfg

        # we perform n = 10 targeted attacks and 1 untargeted attack
        n = 10
        target_labels = dict(itertools.islice(target_labels.items(), n))
        target_idx    = list(target_labels.keys())

        # takes 42.5s for each loss + thresh (all 3 classifiers). In total 42.5*4*6/60=17min for 4 loss, 6 thresh, 3 classifiers
        for stealth_loss in cfg.stealth_losses:
            for d_thr in cfg.d_threshes:
                for classifier_name in cfg.classifier_names:
                    attack_ret_folder  = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                    cam_raw_adv_path   = join(setup_path, 'cam/raw/adv'  , attack_ret_folder)
                    cam_infer_adv_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                    prj_adv_path       = join(setup_path, 'prj/adv'      , attack_ret_folder)

                    # get the true label of the current scene
                    classifier = Classifier(classifier_name, device, cfg.device_ids, fix_params=True, sort_results=dl_based)
                    with torch.no_grad():
                        _, p, pred_idx = classifier(cam_scene, cp_sz)
                    true_idx   = pred_idx[0, 0] if dl_based else p.argmax()  # true label index of the scene given by the current classifier
                    true_label = imagenet_labels[true_idx]

                    print(f'\n-------------------- [{attacker_name}] attacking [{classifier_name}], original prediction: ({true_label}, '
                          f'p={p.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')

                    # untargeted attack
                    targeted_attack = False
                    print(f'[Untargeted] attacking [{classifier_name}]...')

                    if attacker_name == 'SPAA':
                        cam_infer_adv_untar, prj_adv_untar = spaa(model, classifier, imagenet_labels, [true_idx], targeted_attack, cam_scene, d_thr, stealth_loss, cfg.device, setup_info)
                    elif attacker_name == 'PerC-AL+CompenNet++':
                        cam_infer_adv_untar, prj_adv_untar = perc_al_compennet_pp(model, classifier, imagenet_labels, [true_idx], targeted_attack, cam_scene, d_thr, cfg.device, setup_info)
                    elif attacker_name == 'One-pixel_DE':
                        ret, prj_adv_untar, cam_raw_adv_untar = one_pixel_de(im_prj_org, classifier, targeted_attack, target_idx=true_idx,
                                                                             pixel_count=1, pixel_size=41, maxiter=4, popsize=50, verbose=True,
                                                                             true_label=true_label)

                    # targeted attack (batched)
                    targeted_attack = True
                    v = 7  # we only show one adversarial target in the console, v is the index
                    if attacker_name == 'SPAA':
                        print(f'\n[ Targeted ] attacking [{classifier_name}], target: ({imagenet_labels[target_idx[v]]})...')
                        cam_infer_adv_tar, prj_adv_tar = spaa(model, classifier, imagenet_labels, target_idx, targeted_attack, cam_scene, d_thr, stealth_loss, cfg.device, setup_info)
                    elif attacker_name == 'PerC-AL+CompenNet++':
                        print(f'\n[ Targeted ] attacking [{classifier_name}], target: ({imagenet_labels[target_idx[v]]})...')
                        cam_infer_adv_tar, prj_adv_tar = perc_al_compennet_pp(model, classifier, imagenet_labels, target_idx, targeted_attack, cam_scene, d_thr, cfg.device, setup_info)
                    elif attacker_name == 'One-pixel_DE':
                        for i in range(n):
                            print(f'\n[ Targeted ] attacking [{classifier_name}], target: ({imagenet_labels[target_idx[i]]})...')

                            # use a smaller pop size to save time, the result of popsize=50 is almost the same
                            ret, prj_adv_tar, cam_raw_adv_tar = one_pixel_de(im_prj_org, classifier, targeted_attack, target_idx=target_idx[i],
                                                                             pixel_count=1, pixel_size=41, maxiter=4, popsize=10, verbose=True,
                                                                             true_label=true_label)
                            # save adversarial projections and real cam-captured ones
                            ut.save_imgs(expand_4d(cam_raw_adv_tar), cam_raw_adv_path, idx = i)
                            ut.save_imgs(expand_4d(prj_adv_tar), prj_adv_path, idx = i)

                    # save adversarial projections and inferred/real cam-captured ones
                    if dl_based:
                        ut.save_imgs(expand_4d(torch.cat((cam_infer_adv_tar, cam_infer_adv_untar), 0)), cam_infer_adv_path)
                        ut.save_imgs(expand_4d(torch.cat((prj_adv_tar      , prj_adv_untar), 0)), prj_adv_path)
                    else:
                        ut.save_imgs(expand_4d(cam_raw_adv_untar), cam_raw_adv_path, idx = n)
                        ut.save_imgs(expand_4d(prj_adv_untar), prj_adv_path, idx = n)

        if dl_based:
            print(f'\nThe next step is to project and capture [{attacker_name}] generated adversarial projections in {join(setup_path, "prj/adv", attacker_cfg_str)}')
        else:
            print(f'\nThe next step is to inspect the camera-captured adversarial projections in {join(setup_path, "cam/raw/adv", attacker_cfg_str)}')
    return cfg


def project_capture_real_attack(cfg):
    attacker_name = cfg.attacker_name
    assert attacker_name in ['SPAA', 'PerC-AL+CompenNet++'], f'{attacker_name} not supported, One-pixel_DE does not use this function!'
    assert len(cfg.setup_list) == 1, f'The current attacker cfg contains multiple/or no setup_names in setup_list, it should have exactly one setup_name!'

    setup_path = join(cfg.data_root, 'setups', cfg.setup_list[0])
    setup_info = load_setup_info(setup_path)

    for stealth_loss in cfg.stealth_losses:
        for d_thr in cfg.d_threshes:
            for classifier_name in cfg.classifier_names:
                attacker_cfg_str  = to_attacker_cfg_str(cfg.attacker_name)[0]
                attack_ret_folder = join(attacker_cfg_str, stealth_loss  , str(d_thr) , classifier_name)
                prj_input_path    = join(setup_path      , 'prj/adv'    , attack_ret_folder)
                cam_cap_path      = join(setup_path      , 'cam/raw/adv', attack_ret_folder)
                ut.project_capture_data(prj_input_path, cam_cap_path, setup_info)


def get_attacker_cfg(attacker_name, data_root, setup_list, device_ids=[0], load_pretrained=False, plot_on=True):
    # default projector-based attacker configs
    cfg_default = DictConfig({})
    cfg_default.attacker_name       = attacker_name
    cfg_default.classifier_names = ['inception_v3', 'resnet18', 'vgg16']
    cfg_default.data_root           = data_root
    cfg_default.setup_list          = setup_list
    cfg_default.device              = 'cuda'
    cfg_default.device_ids          = device_ids
    cfg_default.load_pretrained     = load_pretrained
    cfg_default.plot_on             = plot_on

    if attacker_name == 'SPAA':
        # cfg_default.stealth_losses = ['caml2', 'camdE', 'camdE_caml2' , 'camdE_caml2_prjl2']
        cfg_default.stealth_losses   = ['caml2', 'camdE', 'camdE_caml2']
        cfg_default.d_threshes       = [5, 7, 9, 11]
    elif attacker_name== 'PerC-AL+CompenNet++':
        cfg_default.stealth_losses   = ['camdE']
        cfg_default.d_threshes       = [11]
    elif attacker_name == 'One-pixel_DE':
        cfg_default.stealth_losses   = ['-']
        cfg_default.d_threshes       = ['-']

    return cfg_default


def to_attacker_cfg_str(attacker_name):
    assert attacker_name in ['SPAA', 'PerC-AL+CompenNet++', 'One-pixel_DE'], f'{attacker_name} not supported!'

    if attacker_name   == 'SPAA':
        model_cfg        = get_model_train_cfg(model_list=['PCNet'], single=True)
        model_cfg_str    = f'{model_cfg.model_name}_{model_cfg.loss}_{model_cfg.num_train}_{model_cfg.batch_size}_{model_cfg.max_iters}'
        attacker_cfg_str = f'{attacker_name}_{model_cfg_str}'
    elif attacker_name == 'PerC-AL+CompenNet++':
        model_cfg        = get_model_train_cfg(model_list=['CompenNet++'], single=True)
        model_cfg_str    = f'{model_cfg.model_name}_{model_cfg.loss}_{model_cfg.num_train}_{model_cfg.batch_size}_{model_cfg.max_iters}'
        attacker_cfg_str = f'{attacker_name}_{model_cfg.loss}_{model_cfg.num_train}_{model_cfg.batch_size}_{model_cfg.max_iters}'
    elif attacker_name == 'One-pixel_DE':
        attacker_cfg_str = f'{attacker_name}'
        model_cfg_str    = None
    return attacker_cfg_str, model_cfg_str


def spaa(pcnet, classifier, imagenet_labels, target_idx, targeted, cam_scene, d_thr, stealth_loss, device, setup_info):
    """
    Stealthy Projector-based Adversarial Attack (SPAA)
    :param pcnet:
    :param classifier:
    :param imagenet_labels:
    :param target_idx:
    :param targeted:
    :param cam_scene:
    :param d_thr: SPAA Algorithm 1's d_thr: threshold for L2 perturbation size
    :param stealth_loss:
    :param device:
    :param setup_info:
    :return:
    """
    device = torch.device(device)
    num_target = len(target_idx)
    cp_sz = setup_info['classifier_crop_sz']

    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1).to(device)

    # projector input image
    im_gray = setup_info['prj_brightness'] * torch.ones(num_target, 3, *setup_info['prj_im_sz']).to(device)  # TODO: cam_train.mean() may be better?
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True

    # [debug] we perform batched targeted attacks, and we only show one adversarial target in console, v is the index
    v = 7 if targeted else 0

    # learning rates
    adv_lr = 2  # SPAA Algorithm 1's \beta_1: step size in minimizing adversarial loss
    col_lr = 1  # SPAA Algorithm 1's \beta_2: step size in minimizing stealthiness loss

    # loss weights, lower adv_w or larger color loss weights reduce success rates but make prj_adv more imperceptible
    # SPAA Eq. 9: adv_w=1, prjl2_w=0, caml2_w=1, camdE_w=0, other combinations are shown in supplementary Sec. 3.2. Different stealthiness loss functions
    adv_w   = 1                                # adversarial loss weights
    prjl2_w = 0.1 if 'prjl2' in stealth_loss else 0    # projector input image l2 loss weights, SPAA paper prjl2_w=0
    caml2_w = 1   if 'caml2' in stealth_loss else 0    # camera captured image l2 loss weights
    camdE_w = 1   if 'camdE' in stealth_loss else 0    # camera captured image deltaE loss weights

    # SPAA Algorithm 1's pthr: threshold for adversarial confidence
    # lower it when SPAA has a good quality, otherwise increase (can get lower class_loss)
    p_thresh = 0.9  # if too high, the attack may not reach it and the output prj_adv is not perturbed, thus is all gray

    # iterative refine the input, SPAA Algorithm 1's K:number of iterations
    iters = 50  # TODO: improve it, we can early stop when attack requirements are met

    prj_adv_best   = prj_adv.clone()
    cam_infer_best = cam_scene.repeat(prj_adv_best.shape[0], 1, 1, 1)
    col_loss_best  = 1e6 * torch.ones(prj_adv_best.shape[0]).to(device)

    for i in range(0, iters):
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)
        raw_score, p, idx = classifier(cam_infer, cp_sz)

        # adversarial loss
        if targeted:
            adv_loss = adv_w * (-raw_score[torch.arange(num_target), target_idx]).mean()
        else:
            adv_loss = adv_w * (raw_score[torch.arange(num_target) , target_idx]).mean()

        # stealthiness loss: prj adversarial pattern should look like im_gray (not used in SPAA)
        prjl2           = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)            # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch  = prjl2_w * prjl2

        # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
        caml2           = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch += caml2_w * caml2

        # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
        camdE           = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(1).mean(1)
        col_loss_batch += camdE_w * camdE

        # average stealthiness (color) losses
        col_loss        = col_loss_batch.mean()

        # mask adversarial confidences that are higher than p_thresh
        mask_high_conf = p[:, 0] > p_thresh
        mask_high_pert = (caml2 * 255 > d_thr).detach().cpu().numpy()

        # alternating between the adversarial loss and the stealthiness (color) loss
        if targeted:
            mask_succ_adv = idx[:, 0] == target_idx
            mask_best_adv = mask_succ_adv & mask_high_conf & mask_high_pert
        else:
            mask_succ_adv = idx[:, 0] != target_idx
            mask_best_adv = mask_succ_adv & mask_high_pert

        # if not successfully attacked, perturb prj_adv toward class_grad
        adv_loss.backward(retain_graph=True)
        adv_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
        prj_adv.data[~mask_best_adv] -= adv_lr * (adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[~mask_best_adv]

        # if successfully attacked, perturb image toward color_grad
        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
        prj_adv.data[mask_best_adv] -= col_lr * (col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[mask_best_adv]

        # keep the best (smallest color loss and successfully attacked ones)
        mask_best_color               = (col_loss_batch < col_loss_best).detach().cpu().numpy()
        mask_best                     = mask_best_color * mask_best_adv
        col_loss_best[mask_best]      = col_loss_batch.data[mask_best].clone()

        # make sure successful adversarial attacks first
        prj_adv_best[mask_succ_adv]   = prj_adv[mask_succ_adv].clone()
        cam_infer_best[mask_succ_adv] = cam_infer[mask_succ_adv].clone()

        # then try to set the best
        prj_adv_best[mask_best]       = prj_adv[mask_best].clone()
        cam_infer_best[mask_best]     = cam_infer[mask_best].clone()

        if i % 30 == 0 or i == iters - 1:
            # lr *= 0.2 # drop lr
            print(f'adv_loss = {adv_loss.item():<9.4f} | col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                  f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {p[v, 0]:.4f} '
                  f'| y = {idx[v, 0]:3d} ({imagenet_labels[idx[v, 0].item()]})')

    # clamp to [0, 1]
    prj_adv_best = torch.clamp(prj_adv_best, 0, 1)  # this inplace opt cannot be used in the for loops above

    return cam_infer_best, prj_adv_best


def perc_al_compennet_pp(compennet_pp, classifier, imgnet_labels, target_idx, targeted, cam_scene, d_thr, device, setup_info):
    # PerC-AL+CompenNet++. A two step based attacker.
    device = torch.device(device)
    num_target = len(target_idx)
    cp_sz = setup_info['classifier_crop_sz']

    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1)

    # 1. Digital attack using PerC-AL;
    confidence = 0 if targeted else 40
    attacker = PerC_AL(device=device, max_iterations=50, alpha_l_init=1, alpha_c_init=0.5, confidence=confidence)
    cam_infer_best = attacker.adversary_projector(classifier, cam_scene_batch, labels=torch.tensor(target_idx).to(device), imagenet_labels=imgnet_labels, d_thr=d_thr, targeted=targeted, cp_sz=cp_sz)

    # 2. Use CompenNet++ to compensate digital adversarial images
    prj_adv_best = compennet_pp(cam_infer_best, cam_scene_batch)

    return cam_infer_best, prj_adv_best


def attack_results(ret, t, imgnet_labels, im_gray, prj_adv, cam_scene, cam_infer, cam_real, prj_im_sz, cp_sz):
    # compute projector-based attack stats and create a result montage

    with torch.no_grad():
        # crop
        cam_scene_cp   = cc(cam_scene.squeeze(), cp_sz)
        cam_real_t_cp  = cc(cam_real[t]        , cp_sz)
        cam_infer_t_cp = cc(cam_infer[t]       , cp_sz)

        # resize to prj_im_sz
        cam_scene_cp_rz   = resize(cam_scene_cp  , tuple(prj_im_sz))
        cam_real_t_cp_rz  = resize(cam_real_t_cp , tuple(prj_im_sz))
        cam_infer_t_cp_rz = resize(cam_infer_t_cp, tuple(prj_im_sz))

        # calculate normalized perturbation for pseudocolor visualization
        cam_real_diff = torch.abs(cam_real_t_cp_rz - cam_scene_cp_rz)
        cam_real_diff = (cam_real_diff - cam_real_diff.min()) / (cam_real_diff.max() - cam_real_diff.min())

        # to pseudo color
        cam_real_diff_color = cv.applyColorMap(np.uint8(cam_real_diff.cpu().numpy().mean(0) * 255), cv.COLORMAP_JET)
        cam_real_diff_color = (torch.Tensor(cv.cvtColor(cam_real_diff_color, cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255).to(cam_scene.device)

        # create result montage
        im = make_grid(torch.stack((cam_scene_cp_rz, prj_adv[t], cam_infer_t_cp_rz, cam_real_t_cp_rz, cam_real_diff_color), 0), nrow=5, padding=5,
                       pad_value=1)

        # calculate stats on cropped image
        prj_l2  = ut.l2_norm(prj_adv[t]    , im_gray)  # mean L2 norm, consistent with Zhao_CVPR_20
        pred_l2 = ut.l2_norm(cam_infer_t_cp, cam_scene_cp)
        real_l2 = ut.l2_norm(cam_real_t_cp , cam_scene_cp)

        # add text
        im = expand_boarder(im, (0, 26, 0, 0))
        im = insert_text(im, f'Cam-captured scene ({t})', (70, 0), 14)
        im = insert_text(im, f'{imgnet_labels[ret["scene"][2][0, 0]]} ({ret["scene"][1][0, 0]:.2f})', (5, 14), 14)

        im = insert_text(im, 'Model inferred adversarial projection', (280, 0), 14)
        im = insert_text(im, f'L2={prj_l2:.2f}', (370, 14), 14)

        im = insert_text(im, 'Model inferred cam-captured projection', (530, 0), 14)
        im = insert_text(im, f'{imgnet_labels[ret["infer"][2][t, 0]]} ({ret["infer"][1][t, 0]:.2f})', (530, 14), 14)
        # im = insertText(im, 'SSIM:{:.2f}'.format(pred_ssim), (715, 14), 14)
        im = insert_text(im, f'L2={pred_l2:.2f}', (720, 14), 14)

        im = insert_text(im, 'Real cam-captured projection', (820, 0), 14)
        im = insert_text(im, f'{imgnet_labels[ret["real"][2][t, 0]]} ({ret["real"][1][t, 0]:.2f})', (790, 14), 14)
        # im = insertText(im, 'SSIM:{:.2f}'.format(real_ssim), (975, 14), 14)
        im = insert_text(im, 'L2={:.2f}'.format(real_l2), (980, 14), 14)

        im = insert_text(im, 'Normalized difference, i.e., 4th-1st', (1070, 0), 14)
        # vfs(im)

    return im


def summarize_single_attacker(attacker_name, data_root, setup_list, device='cuda', device_ids=[0]):
    # given the attacker_name and setup_list, summarize all attacks, create stats.txt/xls and montages
    assert attacker_name in ['SPAA', 'PerC-AL+CompenNet++', 'One-pixel_DE'], f'{attacker_name} not supported!'

    # set PyTorch device to GPU
    device = torch.device(device)
    attacker_cfg_str, model_cfg_str = to_attacker_cfg_str(attacker_name)
    dl_based = attacker_name in ['SPAA', 'PerC-AL+CompenNet++']

    # ImageNet and targeted attack labels
    imagenet_labels = load_imagenet_labels(join(data_root, 'imagenet1000_clsidx_to_labels.txt'))
    target_labels   = load_imagenet_labels(join(data_root, 'imagenet10_clsidx_to_labels.txt'))  # targeted attack labels

    # we perform n = 10 targeted attacks and 1 untargeted attack
    n = 10  # modify n for debug
    target_labels = dict(itertools.islice(target_labels.items(), n))
    target_idx    = list(target_labels.keys())

    # attack results table
    phase   = ['Valid', 'prj', 'infer', 'real']
    metrics = ['PSNR', 'RMSE', 'SSIM', 'L2', 'Linf', 'dE']
    columns = ['Setup', 'Attacker', 'Stealth_loss', 'd_thr', 'Classifier', 'T.top-1_infer', 'T.top-5_infer', 'T.top-1_real', 'T.top-5_real',
               'U.top-1_infer', 'U.top-1_real'] + [phase[0] + '_' + y for y in metrics] +\
              ['T.' + x + '_' + y for x in phase[1:] for y in metrics] + ['U.' + x + '_' + y for x in phase[1:] for y in metrics] + \
              ['All.' + x + '_' + y for x in phase[1:] for y in metrics]

    # stealth_losses = ['caml2', 'camdE', 'camdE_caml2', 'camdE_caml2_prjl2', '-']
    stealth_losses   = ['caml2', 'camdE', 'camdE_caml2', '-']
    d_threshes       = [5, 7, 9, 11, '-']
    classifier_names = ['inception_v3', 'resnet18', 'vgg16']

    for setup_name in setup_list:
        setup_path = join(data_root, 'setups', setup_name)
        print(f'\nCalculating stats of [{attacker_name}] on [{setup_path}]')
        table = pd.DataFrame(columns=columns)

        # load setup info and images
        setup_info = load_setup_info(setup_path)
        cp_sz = setup_info['classifier_crop_sz']

        # projector illumination
        im_gray = setup_info['prj_brightness'] * torch.ones(1, 3, *setup_info['prj_im_sz']).to(device)

        # load training and validation data
        cam_scene  = ut.torch_imread(join(setup_path, 'cam/raw/ref/img_0002.png')).to(device)

        # calc validation metrics
        if attacker_name == 'SPAA':
            im_infer  = cc(ut.torch_imread_mt(join(setup_path, 'cam/infer/test', model_cfg_str)), cp_sz).to(device)
            im_gt     = cc(ut.torch_imread_mt(join(setup_path, 'cam/raw/test')), cp_sz).to(device)
            valid_ret = calc_img_dists(im_infer, im_gt)
        elif attacker_name == 'PerC-AL+CompenNet++':
            im_infer  = ut.torch_imread_mt(join(setup_path, 'prj/infer/test', model_cfg_str)).to(device)
            im_gt     = ut.torch_imread_mt(join(data_root,  'prj_share/test')).to(device)
            valid_ret = calc_img_dists(im_infer, im_gt)
        elif attacker_name == 'One-pixel_DE':
            valid_ret = [0] * 6

        for stealth_loss in stealth_losses:
            for d_thr in d_threshes:
                for classifier_name in classifier_names:
                    attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                    prj_adv_path   = join(setup_path, 'prj/adv'      , attack_ret_folder)
                    cam_infer_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                    cam_real_path  = join(setup_path, 'cam/raw/adv'  , attack_ret_folder)

                    # check whether all images are captured for results summary
                    dirs_to_check = [prj_adv_path, cam_real_path]
                    skip = False
                    if dl_based:
                        dirs_to_check.append(cam_infer_path)

                    for img_dir in dirs_to_check:
                        if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
                            print(f'No such folder/images: {img_dir}\n'
                                  f'Maybe [{attacker_name}] has no [{join(stealth_loss, str(d_thr), classifier_name)}] attack cfg, or you forget to project and capture.\n')
                            skip = True
                            break
                    if skip:
                        break

                    prj_adv   = ut.torch_imread_mt(prj_adv_path).to(device)
                    cam_real  = ut.torch_imread_mt(cam_real_path).to(device)
                    cam_infer = ut.torch_imread_mt(cam_infer_path).to(device) if dl_based else cam_real

                    ret = {}  # classification result dict
                    with torch.no_grad():
                        classifier = Classifier(classifier_name, device, device_ids, fix_params=True, sort_results=True)
                        ret['scene'] = classifier(cam_scene, cp_sz)
                        ret['infer'] = classifier(cam_infer, cp_sz)
                        ret['real']  = classifier(cam_real , cp_sz)

                    # create the result montage as shown in SPAA main paper Figs. 4-5 and supplementary
                    im_montage = []
                    for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                        im_montage.append(attack_results(ret, t, imagenet_labels, im_gray, prj_adv, cam_scene, cam_infer, cam_real, setup_info['prj_im_sz'], cp_sz))

                    # [debug] show montage in visdom
                    # vfs(torch.stack(im_montage, 0), ncol=1, title=attacker_cfg_str + '_' + stealth_loss + '_' + str(d_thr) + '_' + classifier_name)

                    # save montage
                    montage_path = join(setup_path, 'ret', attack_ret_folder)
                    ut.save_imgs(torch.stack(im_montage, 0), montage_path)

                    # Targeted: top-1 and top-5 success rate
                    # inferred attacks
                    t1_infer = np.count_nonzero(ret['infer'][2][:n, 0] == target_idx) / n
                    t5_infer = np.count_nonzero([target_idx[i] in ret['infer'][2][i, :5] for i in range(n)]) / n

                    # real camera-captured attacks
                    t1_real  = np.count_nonzero(ret['real'][2][:n, 0] == target_idx) / n
                    t5_real  = np.count_nonzero([target_idx[i] in ret['real'][2][i, :5] for i in range(n)]) / n

                    # Untargeted: top-1 success rate
                    true_idx = ret['scene'][2][0, 0]
                    t1_untar_infer = np.count_nonzero(ret['infer'][2][n, 0] != true_idx)
                    t1_untar_real  = np.count_nonzero(ret['real'][2][n, 0]  != true_idx)

                    # calc image similarity metrics
                    table.loc[len(table)] = [
                        setup_name, attacker_cfg_str, stealth_loss, d_thr, classifier_name, t1_infer, t5_infer, t1_real,
                        t5_real, t1_untar_infer, t1_untar_real,
                        # model infer vs GT on the validation data
                        *valid_ret,

                        # targeted [0, n-1]
                        *calc_img_dists(prj_adv[:n], im_gray.expand_as(prj_adv[:n])),  # prj adv
                        *calc_img_dists(cc(cam_infer[:n], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer[:n], cp_sz))),  # cam infer
                        *calc_img_dists(cc(cam_real[:n], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real[:n], cp_sz))),
                        # cam real (the last four columns of SPAA paper Table 1 and supplementary Table 4)

                        # untargeted [n]
                        *calc_img_dists(prj_adv[n, None], im_gray.expand_as(prj_adv[n, None])),  # prj adv
                        *calc_img_dists(cc(cam_infer[n, None], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer[n, None], cp_sz))),  # cam infer
                        *calc_img_dists(cc(cam_real[n, None], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real[n, None], cp_sz))),  # cam real

                        # both targeted and untargeted [0, n].
                        *calc_img_dists(prj_adv, im_gray.expand_as(prj_adv)),  # prj adv
                        *calc_img_dists(cc(cam_infer, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer, cp_sz))),  # cam infer
                        *calc_img_dists(cc(cam_real, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real, cp_sz)))  # cam real
                        # !!! The original SPAA paper showed targeted [0, n-1] stealthiness metrics, and missed untargeted [n] by mistake (although it does not change the paper's conclusion).
                        # Future works use the mean of both targeted and untargeted as the last four columns of main paper Table 1 and supplementary Table 4.
                    ]

        # print results
        print(f'\n-------------------- [{attacker_name}] results on [{setup_name}] --------------------')
        print(table.to_string(index=False, float_format='%.4f'))
        print('-------------------------------------- End of result table ---------------------------\n')
        # print(table.filter(regex = 'top-[0-9]_', axis = 1).to_string(index = False, float_format = '%.2f'))  # columns that only contain success rates
        # print(table.filter(regex = '_L2'    , axis = 1).to_string(index = False, float_format = '%.4f'))  # columns that only contain L2
        # print(table.filter(regex = '_dE'    , axis = 1).to_string(index = False, float_format = '%.4f'))  # columns that only contain dE

        # save stats to files
        ret_path = join(setup_path, 'ret', attacker_cfg_str)
        if not os.path.exists(ret_path): os.makedirs(ret_path)
        table.to_csv(join(ret_path, 'stats.txt'), index=False, float_format='%.4f', sep='\t')
        table.to_excel(join(ret_path, 'stats.xlsx'), float_format='%.4f', index=False)
    return table


def summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=False):
    """
    given attacker_names and setup_list, summarize all attacks
    :param attacker_names:
    :param data_root:
    :param setup_list:
    :param recreate_stats_and_imgs: when False, only gather all existing stats.txt of all setups and create a pivot table [setup/pivot_table_all.xlsx]
    :return:
    """
    table = []

    for setup_name in tqdm(setup_list):
        setup_path = join(data_root, 'setups', setup_name)
        for attacker_name in attacker_names:
            attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
            ret_path = join(setup_path, 'ret', attacker_cfg_str)
            print(f'Gathering stats of {ret_path}')

            # (time-consuming) recreate stats.txt, stats.xls and images in [ret] folder for each setup
            if recreate_stats_and_imgs:
                summarize_single_attacker(attacker_name=attacker_name, data_root=data_root, setup_list=[setup_name])
            table.append(pd.read_csv(join(ret_path, 'stats.txt'), index_col=None, header=0, sep='\t'))

    table = pd.concat(table, axis=0, ignore_index=True)

    # pivot_table is supplementary Table 2, and SPAA paper's Table 1 is its subset
    pivot_table = pd.pivot_table(table,
                                 # values=['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'T.real_L2', 'T.real_Linf', 'T.real_dE', 'T.real_SSIM'],
                                 values=['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'T.real_L2', 'T.real_Linf', 'T.real_dE', 'T.real_SSIM', 'All.real_L2', 'All.real_Linf', 'All.real_dE', 'All.real_SSIM'],
                                 index=['Attacker', 'd_thr', 'Stealth_loss', 'Classifier'], aggfunc=np.mean, sort=False)
    pivot_table = pivot_table.sort_index(level=[0, 1], ascending=[False, True]) # to match SPAA Table order

    # save tables
    table.to_csv(join(data_root, 'setups/stats_all.txt'), index=False, float_format='%.4f', sep='\t')
    table.to_excel(join(data_root, 'setups/stats_all.xlsx'), float_format='%.4f', index=False)
    pivot_table.to_excel(join(data_root, 'setups/pivot_table_all.xlsx'), float_format='%.4f', index=True)

    return table, pivot_table

