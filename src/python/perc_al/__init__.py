# Original code obtained from https://github.com/ZhengyuZhao/PerC-Adversarial
# Zhengyu Zhao, Zhuoran Liu, Martha Larson, "Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance", CVPR 2020.
# Extended to work for SPAA and PerC-AL+CompenNet++
# from typing import Tuple, Optional
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
# import torch.optim as optim
from math import pi, cos

from perc_al.differential_color_functions import deltaE, rgb2lab_diff, ciede2000_diff
from tqdm import tqdm

def quantization(x):
    """quantize the continus image tensors into 255 levels (8 bit encoding)"""
    x_quan = torch.round(x * 255) / 255
    return x_quan


class PerC_AL:
    """
    PerC_AL: Alternating Loss of Classification and Color Differences to achieve imperceptibile perturbations with few iterations.
    Parameters
    ----------
    max_iterations : int
        Number of iterations for the optimization.
    alpha_l_init: float
        step size for updating perturbations with respect to classification loss
    alpha_c_init: float
        step size for updating perturbations with respect to perceptual color differences
    confidence : float, optional
        Confidence of the adversary for Carlini's loss, in term of distance between logits.
        Note that this approach only supports confidence setting in an untargeted case
    device : torch.device, optional
        Device on which to perform the adversary.
    """

    def __init__(self,
                 max_iterations: int = 1000,
                 alpha_l_init: float = 1.,
                 # for relatively easy untargeted case, alpha_c_init is adjusted to a smaller value (e.g., 0.1 is used in the paper)
                 alpha_c_init: float = 0.5,
                 confidence: float = 0,
                 device: torch.device = torch.device('cpu')
                 ) -> None:
        self.max_iterations = max_iterations
        self.alpha_l_init = alpha_l_init
        self.alpha_c_init = alpha_c_init
        self.confidence = confidence
        self.device = device

    def adversary(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                  targeted: bool = True) -> torch.Tensor:
        """
        Performs the adversary of the model given the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to fool.
        inputs : torch.Tensor
            Batch of image examples in the range of [0,1].
        labels : torch.Tensor
            Original labels if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted adversary or not.
        Returns
        -------
        torch.Tensor
            Batch of image samples modified to be adversarial
        """

        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        alpha_l_min = self.alpha_l_init / 100
        alpha_c_min = self.alpha_c_init / 10
        multiplier = -1 if targeted else 1

        X_adv_round_best = inputs.clone()
        inputs_LAB = rgb2lab_diff(inputs, self.device)
        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)
        mask_isadv = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        color_l2_delta_bound_best = (torch.ones(batch_size) * 100000).to(self.device)

        if (targeted == False) and self.confidence != 0:
            labels_onehot = torch.zeros(labels.size(0), 1000, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
        if (targeted == True) and self.confidence != 0:
            print('Only support setting confidence in untargeted case!')
            return
        for i in tqdm(range(self.max_iterations)):
            # cosine annealing for alpha_l_init and alpha_c_init
            alpha_c = alpha_c_min + 0.5 * (self.alpha_c_init - alpha_c_min) * (1 + cos(i / self.max_iterations * pi))
            alpha_l = alpha_l_min + 0.5 * (self.alpha_l_init - alpha_l_min) * (1 + cos(i / self.max_iterations * pi))
            loss = multiplier * nn.CrossEntropyLoss(reduction='sum')(model((inputs + delta - 0.5) / 0.5), labels)
            loss.backward()
            grad_a = delta.grad.clone()
            delta.grad.zero_()
            delta.data[~mask_isadv] = delta.data[~mask_isadv] + alpha_l * \
                                      (grad_a.permute(1, 2, 3, 0) / torch.norm(grad_a.view(batch_size, -1), dim=1)).permute(3, 0, 1, 2)[~mask_isadv]
            d_map = ciede2000_diff(inputs_LAB, rgb2lab_diff(inputs + delta, self.device), self.device).unsqueeze(1)
            color_dis = torch.norm(d_map.view(batch_size, -1), dim=1)
            color_loss = color_dis.sum()
            color_loss.backward()
            grad_color = delta.grad.clone()
            delta.grad.zero_()
            delta.data[mask_isadv] = delta.data[mask_isadv] - alpha_c * \
                                     (grad_color.permute(1, 2, 3, 0) / torch.norm(grad_color.view(batch_size, -1), dim=1)).permute(3, 0, 1, 2)[
                                         mask_isadv]

            delta.data = (inputs + delta.data).clamp(0, 1) - inputs
            X_adv_round = quantization(inputs + delta.data)

            if (targeted == False) and self.confidence != 0:
                logits = model((X_adv_round - 0.5) / 0.5)
                real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                other = (logits - labels_infhot).max(1)[0]
                mask_isadv = (real - other) <= -40
            elif self.confidence == 0:
                if targeted:
                    mask_isadv = torch.argmax(model((X_adv_round - 0.5) / 0.5), dim=1) == labels
                else:
                    mask_isadv = torch.argmax(model((X_adv_round - 0.5) / 0.5), dim=1) != labels
            mask_best = (color_dis.data < color_l2_delta_bound_best)
            mask = mask_best * mask_isadv
            color_l2_delta_bound_best[mask] = color_dis.data[mask]
            X_adv_round_best[mask] = X_adv_round[mask]

        return X_adv_round_best

    def adversary_projector(self, classifier: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, imagenet_labels, d_thr,
                            targeted: bool = True, cp_sz=(240, 240)) -> torch.Tensor:
        """
        Modified from PerC-AL's adversary for projector-based attack, also uses p thresh
        Parameters
        ----------
        classifier : nn.Module
            Model to fool.
        inputs : torch.Tensor
            Batch of image examples in the range of [0,1].
        labels : torch.Tensor
            Original labels if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted adversary or not.
        Returns
        -------
        torch.Tensor
            Batch of image samples modified to be adversarial
        """

        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        alpha_l_min = self.alpha_l_init / 100
        alpha_c_min = self.alpha_c_init / 10
        multiplier = -1 if targeted else 1

        p_thresh = 0.9
        device = inputs.device

        # [debug] we perform batched targeted attacks, and we only show one adversarial target in console, v is the index
        v = 7 if targeted else 0

        X_adv_round_best = inputs.clone()
        inputs_LAB = rgb2lab_diff(inputs, self.device)
        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)
        mask_best_adv = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        color_l2_delta_bound_best = (torch.ones(batch_size) * 100000).to(self.device)

        if (targeted == False) and self.confidence != 0:
            labels_onehot = torch.zeros(labels.size(0), 1000, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
        if (targeted == True) and self.confidence != 0:
            print('Only support setting confidence in untargeted case!')
            return
        for i in range(self.max_iterations):
            # cam_infer.data = torch.round(cam_infer.data * 255) / 255  # quantize seems to decrease performance
            raw_score, p, idx = classifier(inputs + delta, cp_sz)

            # cosine annealing for alpha_l_init and alpha_c_init
            alpha_c = alpha_c_min + 0.5 * (self.alpha_c_init - alpha_c_min) * (1 + cos(i / self.max_iterations * pi))
            alpha_l = alpha_l_min + 0.5 * (self.alpha_l_init - alpha_l_min) * (1 + cos(i / self.max_iterations * pi))
            loss = multiplier * nn.CrossEntropyLoss(reduction='sum')(raw_score, labels)
            loss.backward()
            grad_a = delta.grad.clone()
            delta.grad.zero_()

            # pytorch outputs error if we don't do reshape when batch_size = 1
            if batch_size == 1:
                delta.data[~mask_best_adv] = delta.data[~mask_best_adv] + alpha_l * (grad_a.permute(1, 2, 3, 0) / torch.norm(grad_a.reshape(batch_size, -1), dim=1)).permute(3, 0, 1, 2)[~mask_best_adv]
            else:
                delta.data[~mask_best_adv] = delta.data[~mask_best_adv] + alpha_l * (grad_a.permute(1, 2, 3, 0) / torch.norm(grad_a.reshape(batch_size, -1), dim=1)).permute(3, 0, 1, 2)[~mask_best_adv]

            d_map = ciede2000_diff(inputs_LAB, rgb2lab_diff(inputs + delta, self.device), self.device).unsqueeze(1)
            color_dis = torch.norm(d_map.view(batch_size, -1), dim=1)
            color_loss = color_dis.sum()
            color_loss.backward()
            grad_color = delta.grad.clone()
            delta.grad.zero_()

            if batch_size == 1:
                delta.data[mask_best_adv] = delta.data[mask_best_adv] - alpha_c *  (grad_color.permute(1, 2, 3, 0) / torch.norm(grad_color.reshape(batch_size, -1), dim=1)).permute(3, 0, 1, 2)[
                                             mask_best_adv]
            else:
                delta.data[mask_best_adv] = delta.data[mask_best_adv] - alpha_c *  (grad_color.permute(1, 2, 3, 0) / torch.norm(grad_color.reshape(batch_size, -1), dim=1)).permute(3, 0, 1, 2)[
                    mask_best_adv]

            delta.data = (inputs + delta.data).clamp(0, 1) - inputs
            X_adv_round = quantization(inputs + delta.data)

            # added for projector-based attack
            caml2 = torch.norm(delta.detach(), dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
            mask_high_pert = (caml2*255 > d_thr).detach()

            if (targeted == False) and self.confidence != 0:
                # logits = classifier((X_adv_round - 0.5) / 0.5)
                raw_score, p, idx = classifier(X_adv_round, cp_sz)
                mask_high_conf = torch.tensor(p[:, 0] > p_thresh, dtype=torch.bool).to(device)
                real = raw_score.gather(1, labels.unsqueeze(1)).squeeze(1)
                other = (raw_score - labels_infhot).max(1)[0]
                mask_isadv = (real - other) <= -40
                mask_best_adv = mask_isadv & mask_high_pert
            elif self.confidence == 0:
                if targeted:
                    # mask_isadv = torch.argmax(classifier((X_adv_round - 0.5) / 0.5), dim=1) == labels
                    raw_score, p, idx = classifier(X_adv_round, cp_sz)
                    mask_high_conf = torch.tensor(p[:, 0] > p_thresh, dtype=torch.bool).to(device)
                    mask_isadv = (torch.tensor(idx[:, 0]).to(device) == labels)
                    mask_best_adv = mask_isadv & mask_high_conf & mask_high_pert
                else:
                    # mask_isadv = torch.argmax(classifier((X_adv_round - 0.5) / 0.5), dim=1) != labels
                    raw_score, p, idx = classifier(X_adv_round, cp_sz)
                    mask_high_conf = torch.tensor(p[:, 0] > p_thresh, dtype=torch.bool).to(device)
                    mask_isadv = (torch.tensor(idx[:, 0]).to(device) != labels)
                    mask_best_adv = mask_isadv & mask_high_pert

            mask_best_color = (color_dis.data < color_l2_delta_bound_best)
            mask_best = mask_best_color * mask_best_adv
            color_l2_delta_bound_best[mask_best] = color_dis.data[mask_best].clone()

            X_adv_round_best[mask_isadv] = X_adv_round[mask_isadv].clone()
            X_adv_round_best[mask_best] = X_adv_round[mask_best].clone()

            if i % 30 == 0 or i == self.max_iterations - 1:
                # print attack final result
                raw_score, p, idx = classifier(X_adv_round_best, cp_sz)
                adv_loss = multiplier * nn.CrossEntropyLoss(reduction='sum')(raw_score, labels)
                print(f'adv_loss = {adv_loss.item():<9.4f} | camdE = {deltaE(inputs, X_adv_round_best):<9.4f} | p = {p[v, 0]:.4f} '
                      f'| y = {idx[v, 0]:3d} ({imagenet_labels[idx[v, 0].item()]})')

        # print('adv_loss = {:4.4f} | camdE = {:4.4f} | p = {:.4f} | y = {:3d} ({})'.format(adv_loss.item(), deltaE(inputs, X_adv_round_best), p[v, 0], idx[v, 0], imagenet_labels[idx[v, 0].item()]))

        return X_adv_round_best