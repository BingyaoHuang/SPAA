"""
Script to reproduce SPAA (IEEE VR'22) paper/supplementary results on the benchmark dataset (13 setups).

1. We start by setting the training environment to GPU (if any).
2. Setups are listed in 'setup_list', more setups can be found in SPAA/data/setups folder.
3. You need to start visdom first.
4. Run the script by `python reproduce_paper_results.py`, the progress will be updated in console.
5. Upon finish, each setup's results will be saved to [data_root]/setups/[setup_name]/ret, and [data_root]/setups/pivot_table_all.xlsx.
6. To perform real projector-based attacks, refer to `main.py`.

Example:
    python reproduce_paper_results.py

Citation:
    @inproceedings{huang2022spaa,
      title      = {SPAA: Stealthy Projector-based Adversarial Attacks on Deep Image Classifiers},
      booktitle  = {2022 IEEE Conference on Virtual Reality and 3D User Interfaces (VR)},
      author     = {Huang, Bingyao and Ling, Haibin},
      year       = {2022},
      month      = mar,
      pages      = {534--542},
      publisher  = {IEEE},
      address    = {Christchurch, New Zealand},
      doi        = {10.1109/VR51125.2022.00073},
      isbn       = {978-1-66549-617-9}
    }
"""

# %% Set environment
import os
from os.path import join, abspath

# set which GPU(s) to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import print_sys_info, set_torch_reproducibility
from train_network import train_eval_pcnet, train_eval_compennet_pp, get_model_train_cfg
from projector_based_attack import summarize_all_attackers

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), '../../data'))
setup_list = [
    'lotion',
    'soccer',
    'paper_towel',
    'volleyball',
    'backpack',
    'hamper',
    'bucket',
    'coffee_mug',
    'banana',
    'book_jacket',
    'remote_control',
    'mixing_bowl',
    'pillow',
]

# %% reproduce PCNet stats
pcnet_cfg       = get_model_train_cfg(['PCNet', 'PCNet_no_mask_no_rough_d'], data_root, setup_list, load_pretrained=False, plot_on=False)
_, pcnet_ret, _ = train_eval_pcnet(pcnet_cfg)

# %% reproduce CompenNet++ stats
compennet_pp_cfg       = get_model_train_cfg(['CompenNet++'], data_root, setup_list, load_pretrained=False, plot_on=False)
_, compennet_pp_ret, _ = train_eval_compennet_pp(compennet_pp_cfg)

# %% reproduce SPAA paper Table 1 and supplementary Table 4 (takes ~30 min)
attacker_names = ['SPAA', 'PerC-AL+CompenNet++', 'One-pixel_DE']
all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=True)   # recreate stats and images
# all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=False)  # use existing stats and images

print(f'\n------------------ Pivot table of {len(setup_list)} setups in {data_root} ------------------')
print(pivot_table.to_string(index=True, float_format='%.4f'))