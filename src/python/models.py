import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import copy
import pytorch_tps


# CompenNet
class CompenNet(nn.Module):
    def __init__(self):
        super(CompenNet, self).__init__()
        self.name = 'CompenNet'
        self.relu = nn.ReLU()

        # backbone branch
        self.conv1   = nn.Conv2d(3  , 32 , 3, 2, 1)
        self.conv2   = nn.Conv2d(32 , 64 , 3, 2, 1)
        self.conv3   = nn.Conv2d(64 , 128, 3, 1, 1)
        self.conv4   = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5   = nn.Conv2d(256, 128, 3, 1, 1)

        # surface image feature extraction branch
        self.conv1_s = nn.Conv2d(3  , 32 , 3, 2, 1)
        self.conv2_s = nn.Conv2d(32 , 64 , 3, 2, 1)
        self.conv3_s = nn.Conv2d(64 , 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.transConv2 = nn.ConvTranspose2d(64 , 32, 2, 2, 0)
        self.conv6      = nn.Conv2d(32          , 3 , 3, 1, 1)

        # skip layers
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu
        )

        self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 128, 1, 1, 0)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s', None)
        self.register_buffer('res2_s', None)
        self.register_buffer('res3_s', None)
        self.register_buffer('res4_s', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        self.res1_s = self.relu(self.conv1_s(s))
        self.res2_s = self.relu(self.conv2_s(self.res1_s))
        self.res3_s = self.relu(self.conv3_s(self.res2_s))
        self.res4_s = self.relu(self.conv4_s(self.res3_s))

        self.res1_s = self.res1_s.squeeze()
        self.res2_s = self.res2_s.squeeze()
        self.res3_s = self.res3_s.squeeze()
        self.res4_s = self.res4_s.squeeze()

    # x is the input uncompensated image, s is a 1x3x256x256 surface image
    def forward(self, x, s):
        # surface feature extraction
        res1_s = self.relu(self.conv1_s(s)) if self.res1_s is None else self.res1_s
        res2_s = self.relu(self.conv2_s(res1_s)) if self.res2_s is None else self.res2_s
        res3_s = self.relu(self.conv3_s(res2_s)) if self.res3_s is None else self.res3_s
        res4_s = self.relu(self.conv4_s(res3_s)) if self.res4_s is None else self.res4_s

        # backbone
        res1 = self.skipConv1(x)
        x    = self.relu(self.conv1(x) + res1_s)
        res2 = self.skipConv2(x)
        x    = self.relu(self.conv2(x) + res2_s)
        res3 = self.skipConv3(x)
        x    = self.relu(self.conv3(x) + res3_s)
        x    = self.relu(self.conv4(x) + res4_s)
        x    = self.relu(self.conv5(x) + res3)
        x    = self.relu(self.transConv1(x) + res2)
        x    = self.relu(self.transConv2(x))
        x    = torch.clamp(self.relu(self.conv6(x) + res1), max = 1)

        return x


# WarpingNet
class WarpingNet(nn.Module):
    def __init__(self, grid_shape=(6, 6), out_size=(256, 256), with_refine=True):
        super(WarpingNet, self).__init__()
        self.grid_shape  = grid_shape
        self.out_size    = tuple(out_size)
        self.with_refine = with_refine  # becomes WarpingNet w/o refine if set to false
        self.name        = self.__class__.__name__ if with_refine else self.__class__.__name__ + '_without_refine'

        # relu
        self.relu      = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        # final refined grid
        self.register_buffer('fine_grid', None)

        # affine params
        self.affine_mat = nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3))

        # tps params
        self.nctrl  = self.grid_shape[0] * self.grid_shape[1]
        self.nparam = (self.nctrl + 2)
        ctrl_pts    = pytorch_tps.uniform_grid(grid_shape)
        self.register_buffer('ctrl_pts', ctrl_pts.view(-1, 2))
        self.theta  = nn.Parameter(torch.ones((1, self.nparam * 2), dtype=torch.float32).view(-1, self.nparam, 2) * 1e-3)

        # initialization function, first checks the module type,
        def init_normal(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, 1e-4)

        # grid refinement net
        if self.with_refine:
            self.grid_refine_net = nn.Sequential(
                nn.Conv2d(2, 32, 3, 2, 1),
                self.relu,
                nn.Conv2d(32, 64, 3, 2, 1),
                self.relu,
                nn.ConvTranspose2d(64, 32, 2, 2, 0),
                self.relu,
                nn.ConvTranspose2d(32, 2, 2, 2, 0),
                self.leakyRelu
            )
            self.grid_refine_net.apply(init_normal)
        else:
            self.grid_refine_net = None  # WarpingNet w/o refine

    # initialize WarpingNet's affine matrix to the input affine_vec
    def set_affine(self, affine_vec):
        self.affine_mat.data = torch.Tensor(affine_vec).view(-1, 2, 3)

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, x):
        # generate coarse affine and TPS grids
        coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]]), align_corners=True).permute((0, 3, 1, 2))
        coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

        # use TPS grid to sample affine grid
        tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid, align_corners=True)

        # refine TPS grid using grid refinement net and save it to self.fine_grid
        if self.with_refine:
            self.fine_grid = torch.clamp(self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            self.fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))

    def forward(self, x):

        if self.fine_grid is None:
            # not simplified (training/validation)
            # generate coarse affine and TPS grids
            coarse_affine_grid = F.affine_grid(self.affine_mat, torch.Size([1, x.shape[1], x.shape[2], x.shape[3]]), align_corners=True).permute((0, 3, 1, 2))
            coarse_tps_grid = pytorch_tps.tps_grid(self.theta, self.ctrl_pts, (1, x.size()[1]) + self.out_size)

            # use TPS grid to sample affine grid
            tps_grid = F.grid_sample(coarse_affine_grid, coarse_tps_grid, align_corners=True).repeat(x.shape[0], 1, 1, 1)

            # refine TPS grid using grid refinement net and save it to self.fine_grid
            if self.with_refine:
                fine_grid = torch.clamp(self.grid_refine_net(tps_grid) + tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
            else:
                fine_grid = torch.clamp(tps_grid, min=-1, max=1).permute((0, 2, 3, 1))
        else:
            # simplified (testing)
            fine_grid = self.fine_grid.repeat(x.shape[0], 1, 1, 1)

        # warp
        x = F.grid_sample(x, fine_grid, align_corners=True)
        return x


class CompenNetPlusplus(nn.Module):
    # CompenNet++
    def __init__(self, warping_net=None, compen_net=None):
        super(CompenNetPlusplus, self).__init__()
        self.name = 'CompenNet++'

        # initialize from existing models or create new models
        self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.compen_net  = copy.deepcopy(compen_net.module) if compen_net is not None else CompenNet()

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.warping_net.simplify(s)
        self.compen_net.simplify(self.warping_net(s))

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet (both x and s)
        x = self.warping_net(x)
        s = self.warping_net(s)

        # photometric compensation using CompenNet
        x = self.compen_net(x, s)

        return x

class ShadingNetSPAA(nn.Module):
    # Extended from CompenNet
    def __init__(self, use_rough=True):
        super(ShadingNetSPAA, self).__init__()
        self.use_rough = use_rough
        self.name = self.__class__.__name__ if self.use_rough else self.__class__.__name__ + '_no_rough'
        self.relu = nn.ReLU()

        # backbone branch
        self.conv1 = nn.Conv2d(3  , 32 , 3, 2, 1)
        self.conv2 = nn.Conv2d(32 , 64 , 3, 2, 1)
        self.conv3 = nn.Conv2d(64 , 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # surface image feature extraction branch
        num_chan = 6 if self.use_rough else 3
        self.conv1_s = nn.Conv2d(num_chan, 32 , 3, 2, 1)
        self.conv2_s = nn.Conv2d(32      , 64 , 3, 2, 1)
        self.conv3_s = nn.Conv2d(64      , 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128     , 256, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1 , 1)
        self.transConv2 = nn.ConvTranspose2d(64 , 32, 2, 2, 0)
        self.conv6      = nn.Conv2d(32          , 3 , 3, 1, 1)

        # skip layers
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, 0),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu
        )

        self.skipConv2 = nn.Conv2d(32, 64, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s', None)
        self.register_buffer('res2_s', None)
        self.register_buffer('res3_s', None)
        self.register_buffer('res4_s', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s):
        self.res1_s = self.relu(self.conv1_s(s))
        self.res2_s = self.relu(self.conv2_s(self.res1_s))
        self.res3_s = self.relu(self.conv3_s(self.res2_s))
        self.res4_s = self.relu(self.conv4_s(self.res3_s))

        self.res1_s = self.res1_s.squeeze()
        self.res2_s = self.res2_s.squeeze()
        self.res3_s = self.res3_s.squeeze()
        self.res4_s = self.res4_s.squeeze()

    # x is the input uncompensated image, s is a 1x3x256x256 surface image
    def forward(self, x, *argv):
        s = torch.cat(argv, 1)

        # surface feature extraction
        res1_s = self.relu(self.conv1_s(s)) if self.res1_s is None else self.res1_s
        res2_s = self.relu(self.conv2_s(res1_s)) if self.res2_s is None else self.res2_s
        res3_s = self.relu(self.conv3_s(res2_s)) if self.res3_s is None else self.res3_s
        res4_s = self.relu(self.conv4_s(res3_s)) if self.res4_s is None else self.res4_s

        # backbone
        # res1 = self.skipConv1(x)
        res1   = self.skipConv1(argv[0])
        x      = self.relu(self.conv1(x) + res1_s)
        res2   = self.skipConv2(x)
        x      = self.relu(self.conv2(x) + res2_s)
        res3   = self.skipConv3(x)
        x      = self.relu(self.conv3(x) + res3_s)
        x      = self.relu(self.conv4(x) + res4_s)
        x      = self.relu(self.conv5(x) + res3)
        x      = self.relu(self.transConv1(x) + res2)
        x      = self.relu(self.transConv2(x))
        x      = torch.clamp(self.relu(self.conv6(x) + res1), max = 1)

        return x

class PCNet(nn.Module):
    # Project-and-Capture Network for SPAA
    def __init__(self, mask, warping_net=None, shading_net=None, fix_shading_net=False, use_mask=True, use_rough=True):
        super(PCNet, self).__init__()
        self.name      = self.__class__.__name__
        self.use_mask  = use_mask
        self.use_rough = use_rough

        if not self.use_mask:
            self.name += '_no_mask'
        if not self.use_rough:
            self.name += '_no_rough'

        # initialize from existing models or create new models
        self.warping_net = copy.deepcopy(warping_net.module) if warping_net is not None else WarpingNet()
        self.shading_net = copy.deepcopy(shading_net.module) if shading_net is not None else ShadingNetSPAA()

        if self.use_mask:
            self.register_buffer('mask', mask.clone())

        # fix procam net
        for param in self.shading_net.parameters():
            param.requires_grad = not fix_shading_net

    # simplify trained model to a single sampling grid for faster testing
    def simplify(self, s):
        self.warping_net.simplify(s)
        self.shading_net.simplify(self.warping_net(s))

    # s is Bx3x256x256 surface image
    def forward(self, x, s):
        # geometric correction using WarpingNet
        x = self.warping_net(x)

        if self.use_mask:
            x = x * self.mask
        if self.use_rough:
            x = self.shading_net(x, s, x * s)
        else:
            x = self.shading_net(x, s)

        return x