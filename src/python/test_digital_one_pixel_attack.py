"""
Test digital one-pixel attacks
"""
import os
from os.path import join, abspath
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from classifier import Classifier, load_imagenet_labels
from one_pixel_attacker import DigitalOnePixelAttacker
import utils as ut

# %% Digital-based one-pixel attack
# set which GPUs to use
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
ut.reset_rng_seeds(0)
ut.set_torch_reproducibility(False)

# set PyTorch device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load ImageNet labels
imagenet_labels = load_imagenet_labels(join(data_root, 'imagenet1000_clsidx_to_labels.txt'))

# create a classifier
model_name = 'resnet18'  # can be resnet18, vgg16, inception_v3
classifier = Classifier(model_name, device, device_ids, fix_params=True, sort_results=False)

# load an image and perform classification
im = ut.torch_imread(join(data_root, 'sample/anemone_fish.png'))
targeted_attack = False
target_idx = 7 if targeted_attack else 393  # 7 is cock, 393 is anemone fish

# perform digital one-pixel attack
attacker = DigitalOnePixelAttacker(imagenet_labels, (256, 256))
print(f'Performing digital one-pixel {"targeted" if targeted_attack else "untargeted"} attack (label={imagenet_labels[target_idx]}, id={target_idx})')
ret, im_adv = attacker(im, classifier, targeted_attack=targeted_attack, target_idx=target_idx, pixel_count=1, pixel_size=5, maxiter=50,
                       popsize=50, verbose=True, true_label=393)

# show attack results
print(ret.to_string(index=False, float_format='%.4f'))

# show adversarial image and save figure
fig = ut.fs(im_adv, title=f'Digital One-pixel attacked result: {imagenet_labels[ret.pred_idx.item()]} ({ret.pred_p.item():.2f})', facecolor='white')
