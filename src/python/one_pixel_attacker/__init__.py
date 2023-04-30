'''
Digital and projector-based One-pixel adversarial attacker
code modified from: https://github.com/Hyperparticle/one-pixel-attack-keras/blob/92c8506acdcb7807c46dd7404c214ebaaa3d26ad/attack.py
'''

# Helper functions
import os
import torch
import numpy as np
import cv2 as cv
import pandas as pd
from scipy.optimize import differential_evolution
import utils as ut
from img_proc import center_crop as cc
import matplotlib.pyplot as plt

# Projector-based attacks cannot be batched, because we can only project/capture one image at a time.
def perturb_image(x, im, pixel_size):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    # if x.ndim < 2:
    #     x = np.array([x])

    # convert to uint8 type
    im_adv = im.clone() if im.dtype == torch.uint8 else (im * 255).type(torch.uint8)

    # pixel size (must be odd)
    d = pixel_size // 2

    # Make sure to floor the members of xs as int types
    x = x.astype(int)

    # Split x into an array of 5-tuples (perturbation pixels)
    # i.e., [[x,y,r,g,b], ...]
    pixels = np.split(x, len(x) // 5)
    for pixel in pixels:
        # At each pixel's row and col position, assign its rgb value (uint8)
        r, c, *rgb = pixel

        for i in range(r - d, r + d + 1):
            for j in range(c - d, c + d + 1):
                im_adv[:, i, j] = torch.tensor(rgb, dtype=torch.uint8)

    return im_adv


class DigitalOnePixelAttacker:
    def __init__(self, class_names, classifier_crop_sz):
        # Load data and model
        self.class_names = class_names
        self.classifier_crop_sz = classifier_crop_sz

    def perturb_and_predict(self, x, im, classifier, pixel_size):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        im_adv = perturb_image(x, im, pixel_size)
        _, p, _ = classifier(im_adv, self.classifier_crop_sz)

        return p

    def attack_success(self, x, im, target_idx, classifier, pixel_size, targeted_attack=False, verbose=False, true_label=None):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        p = self.perturb_and_predict(x, im, classifier, pixel_size)

        # If the prediction is what we want (misclassification or targeted classification), return True
        if verbose:
            if targeted_attack:
                print(f'Target: {self.class_names[target_idx]:<20} ({p[0, target_idx]:.2f}) | '
                      f'Pred: {self.class_names[p[0].argmax()]:<20} ({p[0].max():.2f}) | '
                      f'GT: {self.class_names[true_label]:<20}')
            else:
                print(f'Untargeted | Pred: {self.class_names[p[0].argmax()]:<20} ({p[0].max():.2f}) | GT: {self.class_names[true_label]:<20}')
        if (targeted_attack and p[0].argmax() == target_idx) or (not targeted_attack and p[0].argmax() != target_idx):
            return True

    def attack(self, im, classifier, targeted_attack=False, target_idx=None, pixel_count=1, pixel_size=1, maxiter=75, popsize=400, verbose=False, true_label=None):

        # pixel size must be odd
        d = pixel_size // 2

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        _, n_rows, n_cols = im.shape  # CxHxW
        bounds = [(d, n_rows - 1 - d), (d, n_cols - 1 - d), (0, 255), (0, 255), (0, 255)] * pixel_count  # bounds are inclusive

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(x):
            p = self.perturb_and_predict(x, im, classifier, pixel_size)

            # This function should always be minimized, so return its complement if needed
            return 1 - p[0, target_idx] if targeted_attack else p[0, target_idx]

        def callback_fn(x, convergence):
            return self.attack_success(x, im, target_idx, classifier, pixel_size, targeted_attack, verbose, true_label)

        # Call Scipy's Implementation of Differential Evolution
        de_ret = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul, recombination=1, atol=-1, callback=callback_fn,
                                        polish=False)

        # Calculate some useful statistics to return from this function
        im_adv = perturb_image(de_ret.x, im, pixel_size).type(torch.float32) / 255
        _, p, _ = classifier(torch.stack((im, im_adv), 0), self.classifier_crop_sz)  # p and idx are sorted, not the original orders.

        true_p, pred_p = p[0].max(), p[1].max()
        true_idx, pred_idx = p[0].argmax(), p[1].argmax()

        if targeted_attack:
            success = pred_idx == target_idx
        else:
            success = pred_idx != true_idx

        cdiff = p[0, target_idx] - p[1, target_idx]

        return pd.DataFrame([[classifier.name, pixel_count, true_idx, pred_idx, success, true_p, pred_p, cdiff]],
                                     columns=['classifier', 'pixel_count', 'true_idx', 'pred_idx', 'success', 'true_p', 'pred_p', 'cdiff']), im_adv

    def __call__(self, im, classifier, targeted_attack=False, target_idx=None, pixel_count=1, pixel_size=1, maxiter=75, popsize=400, verbose=False, true_label=None):
        return self.attack(im, classifier, targeted_attack, target_idx, pixel_count, pixel_size, maxiter, popsize, verbose, true_label)


# Use a projector and a camera to perform projector-based one-pixel attack (One-pixel DE in SPAA paper)
class ProjectorOnePixelAttacker:
    # def __init__(self, class_names, prj_screen_sz, prj_im_sz, cam_raw_sz, cam_crop_sz, cam_im_sz, classifier_crop_sz):
    def __init__(self, class_names, cfg):
        self.prj_screen_sz, self.prj_im_sz                 = cfg['prj_screen_sz'], cfg['prj_im_sz']
        self.cam_raw_sz, self.cam_crop_sz, self.cam_im_sz, = cfg['cam_raw_sz'], cfg['cam_crop_sz'], cfg['cam_im_sz'],
        self.classifier_crop_sz = cfg['classifier_crop_sz']
        self.delay_frames       = cfg['delay_frames']
        self.delay_time         = cfg['delay_time']
        # self.cam_crop_sz      = (min(cam_raw_sz)                        , min(cam_raw_sz))
        self.cam                = ut.init_cam(cfg['cam_raw_sz'])
        self.prj                = ut.init_prj_window(*cfg['prj_screen_sz'], cfg['prj_brightness'])
        self.class_names        = class_names

        self.im_prj_org = None
        self.im_cam_org = None

        # # capture the original object image
        # self.im_prj_org = torch.ones(3, *self.prj_im_sz) * cfg['prj_brightness']
        # self.project(self.im_prj_org, self.delay_time)
        # self.im_cam_org = self.capture(self.delay_frames)

    def project(self, im, delay_time=0.3):
        im_prj = im.permute(1, 2, 0).numpy()  # CxHxW tensor to HxWxC np
        im_prj = im_prj.copy() if im_prj.dtype == np.uint8 else (im_prj * 255).astype(np.uint8)  # here should get np.uint8
        self.prj.set_data(im_prj)
        plt.pause(delay_time)  # a delay time between the project and the capture operations for software sync
        plt.draw()

    def capture(self, delay_frames=13):
        # clear camera buffer
        for j in range(0, delay_frames):
            _, im_cam = self.cam.read()
        # _, im_cam = self.cam.read()
        im_cam = cv.cvtColor(cv.resize(cc(im_cam, self.cam_crop_sz), self.cam_im_sz, interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
        return torch.Tensor(im_cam).permute(2, 0, 1) / 255  # should be torch.float32 within [0, 1]

    def perturb_project_capture(self, x, im, pixel_size):
        # perturb the image with the given pixel(s) x, then project and capture it
        im_prj_adv = perturb_image(x, im, pixel_size)
        self.project(im_prj_adv, self.delay_time)
        im_cam_adv = self.capture(self.delay_frames)

        return im_prj_adv, im_cam_adv

    def step_and_predict(self, x, im, classifier, pixel_size):
        # call perturb_project_capture and classify the projected-and-captured adversarial image
        im_prj_adv, im_cam_adv = self.perturb_project_capture(x, im, pixel_size)

        with torch.no_grad():
            _, p, _ = classifier(im_cam_adv, self.classifier_crop_sz)

        return p

    def attack_success(self, x, im, target_idx, classifier, pixel_size, targeted_attack=False, verbose=False, true_label=None):
        p = self.step_and_predict(x, im, classifier, pixel_size)

        # if the prediction is what we want (misclassification or targeted classification), return True
        if verbose:
            if targeted_attack:
                print(f'Target: {self.class_names[target_idx]:<20} ({p[0, target_idx]:.2f}) | '
                      f'Pred: {self.class_names[p[0].argmax()]:<20} ({p[0].max():.2f}) | '
                      f'GT: {true_label:<15}')
            else:
                print(f'Untargeted | Pred: {self.class_names[p[0].argmax()]:<20} ({p[0].max():.2f}) | GT: {true_label:<15}')

        if (targeted_attack and p[0].argmax() == target_idx) or (not targeted_attack and p[0].argmax() != target_idx):
            return True

    def attack(self, im, classifier, targeted_attack=False, target_idx=None, pixel_count=1, pixel_size=1, maxiter=75, popsize=400, verbose=False, true_label=None):
        """
        :param im: the initial projector image
        """

        # pixel size must be odd
        d = pixel_size // 2

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        _, n_rows, n_cols = im.shape  # CxHxW
        bounds = [(d, n_rows - 1 - d), (d, n_cols - 1 - d), (0, 255), (0, 255), (0, 255)] * pixel_count  # bounds are inclusive

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(x):
            p = self.step_and_predict(x, im, classifier, pixel_size)

            # This function should always be minimized, so return its complement if needed
            return 1 - p[0, target_idx] if targeted_attack else p[0, target_idx]

        def callback_fn(x, convergence):
            return self.attack_success(x, im, target_idx, classifier, pixel_size, targeted_attack, verbose, true_label)

        # Call Scipy's Implementation of Differential Evolution
        de_ret = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul, recombination=1, atol=-1, callback=callback_fn,
                                        polish=False)

        # Calculate some useful statistics to return from this function
        im_prj_adv, im_cam_adv = self.perturb_project_capture(de_ret.x, im, pixel_size)
        # self.cam.release()

        _, p, _ = classifier(torch.stack((cc(self.im_cam_org, self.classifier_crop_sz), cc(im_cam_adv, self.classifier_crop_sz)), 0),
                             self.classifier_crop_sz)  # p and idx are sorted, not in the original orders.

        true_p, pred_p = p[0].max(), p[1].max()
        true_idx, pred_idx = p[0].argmax(), p[1].argmax()

        if targeted_attack:
            success = pred_idx == target_idx
        else:
            success = pred_idx != true_idx

        cdiff = p[0, target_idx] - p[1, target_idx]  # confidence changes of before and after the attack

        return pd.DataFrame([[classifier.name, pixel_count, true_idx, pred_idx, success, true_p, pred_p, cdiff]],
                                     columns=['classifier', 'pixel_count', 'true_idx', 'pred_idx', 'success', 'true_p', 'pred_p',
                                     'cdiff']), im_prj_adv, im_cam_adv

    def __call__(self, im, classifier, targeted_attack=False, target_idx=None, pixel_count=1, pixel_size=1, maxiter=75, popsize=400, verbose=False, true_label=None):
        return self.attack(im, classifier, targeted_attack, target_idx, pixel_count, pixel_size, maxiter, popsize, verbose, true_label)