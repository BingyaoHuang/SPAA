[Dataset] SPAA: Stealthy Projector-based Adversarial Attacks on Deep Image Classifiers
======================================================================================

## Usage

Download and extract this zip to folder ``SPAA/data`` and follow instructions [here][3]. See [paper][1] and [supplementary][2].

## Folder Structure

    ├─prj_share                # projector input images shared by each setup
    │  ├─init                  # for CompenNet initialization
    │  ├─numbers               # test whether the projector and the camera are in sync
    │  ├─test                  # same as CompenNet++'s test data
    │  └─train                 # same as CompenNet++'s train data
    ├─sample
    └─setups                   # parent folder of all setups
        ├─backpack             # a setup's name (better to use ImageNet labels)
        │  ├─cam               # camera related images
        │  │  ├─infer          # model inferred camera-captured images (PCNet and CompenNet++ only)
        │  │  │  ├─adv         # inferred camera-captured adversarial projections
        │  │  │  │  ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000         # PerC-AL+CompenNet++ inferred
        │  │  │  │  │  └─camdE                                        # stealthiness loss, e.g., CIE delta E
        │  │  │  │  │      └─11                                       # L2 perturbation sizes, i.e., d_thr in SPAA paper
        │  │  │  │  │          ├─inception_v3                         # classifier name
        │  │  │  │  │          ├─resnet18
        │  │  │  │  │          └─vgg16
        │  │  │  │  └─SPAA_PCNet_l1+ssim_500_24_2000                  # SPAA inferred
        │  │  │  │      ├─camdE
        │  │  │  │      │  ├─11
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─5
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─7
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  └─9
        │  │  │  │      │      ├─inception_v3
        │  │  │  │      │      ├─resnet18
        │  │  │  │      │      └─vgg16
        │  │  │  │      ├─camdE_caml2                                 # stealthiness loss, e.g., CIE delta E + L2
        │  │  │  │      │  ├─11
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─5
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─7
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  └─9
        │  │  │  │      │      ├─inception_v3
        │  │  │  │      │      ├─resnet18
        │  │  │  │      │      └─vgg16
        │  │  │  │      └─caml2
        │  │  │  │          ├─11
        │  │  │  │          │  ├─inception_v3
        │  │  │  │          │  ├─resnet18
        │  │  │  │          │  └─vgg16
        │  │  │  │          ├─5
        │  │  │  │          │  ├─inception_v3
        │  │  │  │          │  ├─resnet18
        │  │  │  │          │  └─vgg16
        │  │  │  │          ├─7
        │  │  │  │          │  ├─inception_v3
        │  │  │  │          │  ├─resnet18
        │  │  │  │          │  └─vgg16
        │  │  │  │          └─9
        │  │  │  │              ├─inception_v3
        │  │  │  │              ├─resnet18
        │  │  │  │              └─vgg16
        │  │  │  └─test         # model inferred camera-captured test images (PCNet and CompenNet++ only)
        │  │  │      ├─PCNet_l1+ssim_500_24_2000
        │  │  │      └─PCNet_no_mask_no_rough_d_l1+ssim_500_24_2000
        │  │  └─raw             # real camera-captured images
        │  │      ├─adv
        │  │      │  ├─One-pixel_DE
        │  │      │  │  └─-        # One-pixel_DE does not use stealthiness loss
        │  │      │  │      └─-    # One-pixel_DE does not use perturbation sizes threshold
        │  │      │  │          ├─inception_v3
        │  │      │  │          ├─resnet18
        │  │      │  │          └─vgg16
        │  │      │  ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │  │      │  │  └─camdE
        │  │      │  │      └─11
        │  │      │  │          ├─inception_v3
        │  │      │  │          ├─resnet18
        │  │      │  │          └─vgg16
        │  │      │  └─SPAA_PCNet_l1+ssim_500_24_2000
        │  │      │      ├─camdE
        │  │      │      │  ├─11
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─5
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─7
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  └─9
        │  │      │      │      ├─inception_v3
        │  │      │      │      ├─resnet18
        │  │      │      │      └─vgg16
        │  │      │      ├─camdE_caml2
        │  │      │      │  ├─11
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─5
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─7
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  └─9
        │  │      │      │      ├─inception_v3
        │  │      │      │      ├─resnet18
        │  │      │      │      └─vgg16
        │  │      │      └─caml2
        │  │      │          ├─11
        │  │      │          │  ├─inception_v3
        │  │      │          │  ├─resnet18
        │  │      │          │  └─vgg16
        │  │      │          ├─5
        │  │      │          │  ├─inception_v3
        │  │      │          │  ├─resnet18
        │  │      │          │  └─vgg16
        │  │      │          ├─7
        │  │      │          │  ├─inception_v3
        │  │      │          │  ├─resnet18
        │  │      │          │  └─vgg16
        │  │      │          └─9
        │  │      │              ├─inception_v3
        │  │      │              ├─resnet18
        │  │      │              └─vgg16
        │  │      ├─cb          # camera-captured checkerboard images for PCNet direct light extraction (Nayar TOG'06)
        │  │      ├─ref         # camera-captured black/gray/white images for PCNet/CompenNet++ training, where gray is also used as the scene image
        │  │      ├─test        # camera-captured test images for PCNet/CompenNet++ training
        │  │      └─train       # camera-captured train images for PCNet/CompenNet++ training
        │  ├─prj                # projector input images to be projected to the scene for projector-based attacks
        │  │  ├─adv             # adversarial projections generated by the three projector-based attackers
        │  │  │  ├─One-pixel_DE
        │  │  │  │  └─-
        │  │  │  │      └─-
        │  │  │  │          ├─inception_v3
        │  │  │  │          ├─resnet18
        │  │  │  │          └─vgg16
        │  │  │  ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │  │  │  │  └─camdE
        │  │  │  │      └─11
        │  │  │  │          ├─inception_v3
        │  │  │  │          ├─resnet18
        │  │  │  │          └─vgg16
        │  │  │  └─SPAA_PCNet_l1+ssim_500_24_2000
        │  │  │      ├─camdE
        │  │  │      │  ├─11
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─5
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─7
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  └─9
        │  │  │      │      ├─inception_v3
        │  │  │      │      ├─resnet18
        │  │  │      │      └─vgg16
        │  │  │      ├─camdE_caml2
        │  │  │      │  ├─11
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─5
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─7
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  └─9
        │  │  │      │      ├─inception_v3
        │  │  │      │      ├─resnet18
        │  │  │      │      └─vgg16
        │  │  │      └─caml2
        │  │  │          ├─11
        │  │  │          │  ├─inception_v3
        │  │  │          │  ├─resnet18
        │  │  │          │  └─vgg16
        │  │  │          ├─5
        │  │  │          │  ├─inception_v3
        │  │  │          │  ├─resnet18
        │  │  │          │  └─vgg16
        │  │  │          ├─7
        │  │  │          │  ├─inception_v3
        │  │  │          │  ├─resnet18
        │  │  │          │  └─vgg16
        │  │  │          └─9
        │  │  │              ├─inception_v3
        │  │  │              ├─resnet18
        │  │  │              └─vgg16
        │  │  ├─infer      # model inferred projector input images (CompenNet++ only)
        │  │  │  └─test
        │  │  │      └─CompenNet++_l1+ssim_500_24_2000
        │  │  └─raw        # projector input checkerboard and illumination images (created by code according to user's setup_info)
        │  │      ├─cb
        │  │      └─ref
        │  └─ret           # results of real projector-based attacks, including comparison images like SPAA paper's Figs. 4-5 and numbers in Table 1.
        │      ├─One-pixel_DE
        │      │  └─-
        │      │      └─-
        │      │          ├─inception_v3
        │      │          ├─resnet18
        │      │          └─vgg16
        │      ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │      │  └─camdE
        │      │      └─11
        │      │          ├─inception_v3
        │      │          ├─resnet18
        │      │          └─vgg16
        │      └─SPAA_PCNet_l1+ssim_500_24_2000
        │          ├─camdE
        │          │  ├─11
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─5
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─7
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  └─9
        │          │      ├─inception_v3
        │          │      ├─resnet18
        │          │      └─vgg16
        │          ├─camdE_caml2
        │          │  ├─11
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─5
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─7
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  └─9
        │          │      ├─inception_v3
        │          │      ├─resnet18
        │          │      └─vgg16
        │          └─caml2
        │              ├─11
        │              │  ├─inception_v3
        │              │  ├─resnet18
        │              │  └─vgg16
        │              ├─5
        │              │  ├─inception_v3
        │              │  ├─resnet18
        │              │  └─vgg16
        │              ├─7
        │              │  ├─inception_v3
        │              │  ├─resnet18
        │              │  └─vgg16
        │              └─9
        │                  ├─inception_v3
        │                  ├─resnet18
        │                  └─vgg16
        ├─banana     # another setup

## Citation

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

## Acknowledgments

- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
- We thank the authors of the colorful textured sampling images.
- Feel free to open an issue if you have any questions/suggestions/concerns 😁. 

[1]: https://bingyaohuang.github.io/pub/SPAA
[2]: https://bingyaohuang.github.io/pub/SPAA/supp
[3]: https://github.com/BingyaoHuang/SPAA
