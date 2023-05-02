'''
Useful helper functions
'''

import os
from os.path import join, abspath
import sys
import warnings
import platform
import math
import random
import pandas as pd
import skimage.util
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_ssim
import time
import yaml

from perc_al import deltaE
from img_proc import resize, center_crop as cc

from tqdm import tqdm

# avoid pandas console display truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# use qt5agg backend for remote interactive interpreter plot below
import matplotlib as mpl

# backend (if not specified, mpl will automatically decide)
# restart MobaXterm if "Cannot load backend 'Qt5Agg' which requires the 'qt5' interactive framework, as 'headless' is currently running"
# mpl.use('Qt5Agg') # image cannot correctly resize in full screen for init_prj_window()
# mpl.use('QtAgg')  # [not compatible with PyCharm 2023.1] need PySide6, and figure may freeze if run it in PyCharm 2023.1's Python console. Maybe setting PyCharm 'PyQt compatible' to 'pyside6' solve this issue? https://youtrack.jetbrains.com/issue/PY-54194
# mpl.use('TkAgg')  # cannot create stand-alone window for init_prj_window() in Jupyter notebook

# disable toolbar and set background to black for full screen
mpl.rcParams['toolbar'] = 'None'
mpl.rcParams['figure.facecolor'] = 'black'

import matplotlib.pyplot as plt  # restart X11 session if it hangs (MobaXterm in my case)


def init_visdom(server='localhost', port=8097):
    import visdom
    vis = visdom.Visdom(server=server, port=port, use_incoming_socket=False)  # default is 8097
    assert vis.check_connection(), 'Visdom: No connection, start visdom first!'

    # if not check_visdom_open():
    #     # vis = visdom.Visdom(port=8097, use_incoming_socket=False)  # default is 8097
    #     vis = visdom.Visdom(server=server, port=port, use_incoming_socket=False)  # default is 8097
    #     assert vis.check_connection(), 'Visdom: No connection, start visdom first!'
    # else:
    #     vis = visdom.Visdom(server=server, port=port)
    return vis


vis = init_visdom(port=8097)


def reset_rng_seeds(seed):
    # set random number generators' seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_torch_reproducibility(reproducible=False):
    torch.backends.cudnn.deterministic = True if reproducible else False
    torch.backends.cudnn.benchmark = False if reproducible else True


# ------------------------------------------- Image IO  -----------------------------------------------
class SimpleDataset(Dataset):
    """Simple dataset."""

    # Use Pytorch multi-threaded dataloader and opencv to load image faster

    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size

        # img list
        img_list = sorted(os.listdir(data_root))
        if index is not None: img_list = [img_list[x] for x in index]

        self.img_names = [join(self.data_root, name) for name in img_list]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        assert os.path.isfile(img_name), img_name + ' does not exist'
        im = cv.imread(self.img_names[idx])

        # resize image if size is specified
        if self.size is not None:
            im = cv.resize(im, self.size[::-1])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        return im


# read a single image to float tensor CxHxW
def torch_imread(filename):
    return torch.Tensor(cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255


def torch_imread_mt(img_dir, size=None, index=None, gray_scale=False, normalize=False):
    # read images using multi-thread
    # size must be in (h, w) format
    img_dataset = SimpleDataset(img_dir, index=index, size=size)
    # data_loader = DataLoader(img_dataset, batch_size=len(img_dataset), shuffle=False, drop_last=False, num_workers=4) # win may have problem with more than 1 worker
    # num_workers = 4 if sys.platform == 'linux' else 0  # windows may complain when num_workers>0
    num_workers = 0

    data_loader = DataLoader(img_dataset, batch_size=len(img_dataset), shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    for i, imgs in enumerate(data_loader):
        # convert to torch.Tensor
        imgs = imgs.permute((0, 3, 1, 2)).float().div(255)

        if gray_scale:
            imgs = 0.2989 * imgs[:, 0] + 0.5870 * imgs[:, 1] + 0.1140 * imgs[:, 2]  # same as MATLAB rgb2gray and OpenCV cvtColor
            imgs = imgs[:, None]

        # normalize to [-1, 1], may improve model convergence in early training stages.
        if normalize:
            imgs = (imgs - 0.5) / 0.5

        return imgs


# save 4D np.ndarray or torch tensor to image files
def save_imgs(im_4d, path, idx=0):
    if not os.path.exists(path):
        os.makedirs(path)

    if type(im_4d) is torch.Tensor:
        if im_4d.requires_grad:
            im_4d = im_4d.detach()
        if im_4d.device.type == 'cuda':
            imgs = im_4d.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = im_4d.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
    else:
        imgs = im_4d

    # imgs must have a shape of (N, row, col, C)
    if imgs.dtype == 'float32':
        imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    else:
        imgs = imgs[:, :, :, ::-1]  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(i + 1 + idx)
        cv.imwrite(join(path, file_name), imgs[i, :, :, :])  # faster than PIL or scipy


# ------------------------------------------- Plot functions  -----------------------------------------------
def fs(input_data, title=None, cmap='gray', facecolor='black'):
    # figure; show, create a figure and show a tensor or ndarray
    input_data = input_data.squeeze()
    if type(input_data) is np.ndarray:
        im = input_data
    elif type(input_data) is torch.Tensor:
        F_tensor_to_image = torchvision.transforms.ToPILImage()

        if input_data.requires_grad:
            input_data = input_data.detach()

        if input_data.device.type == 'cuda':
            if input_data.ndimension() == 2:
                im = input_data.squeeze().cpu().numpy()
            else:
                im = F_tensor_to_image(input_data.squeeze().cpu())
        else:
            if input_data.ndimension() == 2:
                im = input_data.numpy()
            else:
                im = F_tensor_to_image(input_data.squeeze())

    # remove white paddings
    fig = plt.figure(facecolor=facecolor)
    # fig = plt.figure()
    # fig.canvas.window().statusBar().setVisible(False)

    # display image
    ax = plt.imshow(im, interpolation='bilinear', cmap=cmap)
    ax = plt.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if title is not None:
        plt.title(title, color='red')
        plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0, hspace=0, wspace=0)
    plt.show()
    return fig


def vfs(x, padding=10, title=None, ncol=None):
    # visdom figure show
    nrow = 5 if ncol is None else ncol
    t = title if title is not None else ''

    if x.ndim == 3:
        return vis.image(x, opts=dict(title=t, caption=t))
    elif x.ndim == 4 and x.shape[0] == 1:
        return vis.image(x[0], opts=dict(title=t, caption=t))
    else:
        return vis.images(x, opts=dict(title=t, caption=t), nrow=nrow, padding=padding)


def append_data_point(x, y, win, name, env=None):
    # append a data point to the visdom curve in win
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update='append',
        name=name,
        opts=dict(markers=True, markersize=3)
    )


def vis_pcnet_process(pcnet, prj_input, cam_scene, cam_train, cam_mask, cp_sz):
    device = prj_input.device

    with torch.no_grad():
        # simplify PCNet
        # pcnet.module.simplify(prj_input)
        prj_warp, _ = pcnet.module.warping_net(prj_input)
        prj_warp_masked = torch.where(cam_mask, prj_warp.cpu(), torch.tensor([0.])).to(device)
        cam_rough = prj_warp_masked * cam_scene
        cam_infer, _ = pcnet(prj_input, cam_scene)
        cam_mask_vis = cam_mask.expand_as(cam_scene).float().to(device)

    # im_montage = make_grid(torch.cat((cam_scene, cam_mask_vis, resize(prj_input, cp_sz), resize(prj_warp, cp_sz),
    #                                   prj_warp_masked, cam_rough, cam_infer, cam_train), 0), nrow=4, padding=5, pad_value=1)

    im_montage = make_grid_transposed(torch.cat((cam_scene, cam_mask_vis, resize(prj_input, cp_sz), resize(prj_warp, cp_sz),
                                                 prj_warp_masked, cam_rough, cam_infer, cam_train), 0), nrow=2, padding=5, pad_value=1)
    vfs(im_montage, title='PCNet process (intermediate results)')


def plot_montage(*argv, index=None, win=None, title=None, env=None, grid_w=5, cp_sz=None):
    """
    Plot montage using visdom
    """
    with torch.no_grad():  # just in case
        # compute montage grid size
        if argv[0].shape[0] > grid_w:
            # grid_w = 5
            idx = random.sample(range(argv[0].shape[0]), grid_w) if index is None else index
        else:
            grid_w = argv[0].shape[0]
            # idx = random.sample(range(cam_im.shape[0]), grid_w)
            idx = range(grid_w)

        # resize to (256, 256) for better display
        tile_size = (256, 256)
        im_resize = torch.empty((len(argv) * grid_w, argv[0].shape[1]) + tile_size)
        i = 0
        for im in argv:
            if im.shape[2] != tile_size[0] or im.shape[3] != tile_size[1]:
                if cp_sz is not None:
                    im_resize[i:i + grid_w] = F.interpolate(cc(im[idx, :, :, :], cp_sz), tile_size)
                else:
                    im_resize[i:i + grid_w] = F.interpolate(im[idx, :, :, :], tile_size)
            else:
                if cp_sz is not None:
                    im_resize[i:i + grid_w] = F.interpolate(cc(im[idx, :, :, :], cp_sz), tile_size)
                else:
                    im_resize[i:i + grid_w] = im[idx, :, :, :]
            i += grid_w

        # title
        # plot_opts = dict(title=title, caption=title, font=dict(size=18), store_history=False)
        plot_opts = dict(title=title, caption=title, font=dict(size=18), tore_history=False)

        # plot montage to existing win, otherwise create a new win
        im_montage = torchvision.utils.make_grid(im_resize, nrow=grid_w, padding=10, pad_value=1)

        win = vis.image(im_montage, win=win, opts=plot_opts, env=env)
    return win


def montage(im_in, grid_shape=None, padding_width=5, fill=(1, 1, 1), multichannel=True):
    # create an image montage from a (row, col, C, N) np.ndarray or (N, row, col, C) tensor
    if type(im_in) is np.ndarray:
        assert im_in.ndim == 4, 'requires a 4-D array with shape (row, col, C, N)'
        im = im_in.transpose(3, 0, 1, 2)

    elif type(im_in) is torch.Tensor:
        assert im_in.ndimension() == 4, 'requires a 4-D tensor with shape (N, C, row, col)'

        if im_in.device.type == 'cuda':
            im_in = im_in.cpu()
        if im_in.requires_grad:
            im_in = im_in.detach()
        im = im_in.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    if grid_shape is None:
        num_rows = math.ceil(math.sqrt(im.shape[0]))
        num_cols = math.ceil(im.shape[0] / num_rows)
        grid_shape = (num_rows, num_cols)
    else:
        num_rows = grid_shape[0]
        num_cols = grid_shape[1]
        if num_rows == -1:
            grid_shape = (im.shape[0] / num_cols, num_cols)
        elif num_cols == -1:
            grid_shape = (num_rows, im.shape[0] / num_rows)

    im_out = skimage.util.montage(im, rescale_intensity=False, multichannel=multichannel, padding_width=padding_width, fill=fill,
                                  grid_shape=grid_shape)

    return im_out


def make_grid_transposed(tensor, nrow=8, padding=2, normalize=False, irange=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    # from torchvision.utils.make_grid, but row and col are switched
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        irange (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if irange is not None:
            assert isinstance(irange, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, irange)
        else:
            norm_range(tensor, irange)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    ymaps = min(nrow, nmaps)
    xmaps = int(math.ceil(float(nmaps) / ymaps))

    width, height = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0

    for x in range(xmaps):
        for y in range(ymaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


# ------------------------------------------- image distances -----------------------------------------------
def calc_img_dists(x, y):
    # Compute PSNR/RMSE/SSIM/L2/L_inf/deltaE
    with torch.no_grad():
        return psnr(x, y), rmse(x, y), ssim(x, y), l2_norm(x, y), linf_norm(x, y), deltaE(x, y)


# compute PSNR
def psnr(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return 10 * math.log10(1 / l2_fun(x, y))


# compute RMSE
def rmse(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return math.sqrt(l2_fun(x, y).item() * 3)  # only works for RGB, for grayscale, don't multiply by 3


# compute SSIM
def ssim(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        return pytorch_ssim.ssim(x, y).item()


# compute L2 norm (mean L2 norm, consistent with Zhao_CVPR_20)
def l2_norm(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        diff = x - y
        if diff.ndim == 3:
            ret = torch.norm(diff, p=2, dim=0).mean().item() * 255
        if diff.ndim == 4:
            ret = torch.norm(diff, p=2, dim=1).mean().item() * 255
        return ret


# compute L_inf norm
def linf_norm(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        diff = x - y
        if diff.ndim == 3:
            ret = torch.norm(diff, p=float('inf'), dim=0).mean().item() * 255
        if diff.ndim == 4:
            ret = torch.norm(diff, p=float('inf'), dim=1).mean().item() * 255
        return ret


# convert l2 norm to mse
def l2_norm_to_mse(x, num_chan):
    # x = torch.norm(a, p=2, dim=1), where a shape is BxCxHxW, x shape is BxHxW, num_chan: number of channels (C)
    return (x ** 2).mean() / num_chan


# ------------------------------------------- Data capture -----------------------------------------------
def init_prj_window(prj_w, prj_h, val, offset=(3900, -300)):
    """
    Initialize the projector window using plt
    :param prj_w:
    :param prj_h:
    :param val:
    :param offset: move the projector window by an offset in (x, y) format
    :return:
    """
    # initial image
    im = np.ones((prj_h, prj_w, 3), np.uint8) * val
    disp_size = min(prj_h, prj_w)
    im = cv.resize(im, (disp_size, disp_size))

    # create figure and move to projector screen and set to full screen
    fig = plt.figure()

    # uncheck pycharm scientific mode when you encounter error "AttributeError: 'FigureCanvasInterAgg' object has no attribute 'window'"
    fig.canvas.window().statusBar().setVisible(False)  # (QtAgg only)

    ax = plt.imshow(im, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    mng = plt.get_current_fig_manager()
    mng.window.setGeometry(*offset, prj_w, prj_h)  # change the offsets according to your setup (QtAgg only)
    plt.pause(0.02)  # !!! MUST PAUSE, OTHERWISE FIGURE MOVES TO THE PRIMARY SCREEN
    mng.full_screen_toggle()  # to full screen (TkAgg may not work well, and resets window position to primary screen)

    fig.show()

    return ax


def init_cam(cam_raw_sz=None):
    if sys.platform == 'win32':
        cam = cv.VideoCapture(0, cv.CAP_DSHOW)  # windows only to get rid of the annoying warning
    else:
        cam = cv.VideoCapture(0)

    if cam_raw_sz is not None:
        cam.set(cv.CAP_PROP_FRAME_WIDTH, cam_raw_sz[0])
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, cam_raw_sz[1])
    cam.set(cv.CAP_PROP_BUFFERSIZE, 1)  # set the buffer size to 1 to avoid delayed frames
    cam.set(cv.CAP_PROP_FPS, 60)  # set the max frame rate (cannot be larger than cam's physical fps) to avoid delayed frames
    time.sleep(2)

    if not cam.isOpened():
        print("Cannot open camera")
    return cam


def preview_cam(cam_raw_sz=None, cam_crop_sz=None):
    cam = init_cam(cam_raw_sz)

    while True:
        # Capture frame-by-frame
        ret, im_cam = cam.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        im = cc(im_cam, cam_crop_sz) if cam_crop_sz is not None else im_cam

        # Display the resulting frame
        cv.imshow('Camera preview (press q to exit)', im)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv.destroyAllWindows()


def project_capture_data(prj_input_path, cam_cap_path, setup_info):
    print(f'Projecting {prj_input_path} and \ncapturing to {cam_cap_path}')
    # all sz are in (w, h) format
    prj_screen_sz, prj_offset, cam_raw_sz, cam_crop_sz, cam_im_sz, delay_frames, delay_time = setup_info['prj_screen_sz'], setup_info['prj_offset'], setup_info['cam_raw_sz'], setup_info[
        'cam_crop_sz'], setup_info['cam_im_sz'], setup_info['delay_frames'], setup_info['delay_time']

    if not os.path.exists(cam_cap_path):
        os.makedirs(cam_cap_path)

    # load projector input images
    im_prj = np.uint8(torch_imread_mt(prj_input_path).permute(2, 3, 1, 0) * 255)
    prj_im_aspect = im_prj.shape[1] / im_prj.shape[0]
    prj_screen_aspect = prj_screen_sz[0] / prj_screen_sz[1]

    # check aspect ratio
    if math.isclose(prj_im_aspect, prj_screen_aspect, abs_tol=1e-3):
        warnings.warn(f'The projector input image aspect ratio {prj_im_aspect} is different from the screen aspect ratio {prj_screen_aspect}, '
                      f'image will be resized to center fill the screen')

    # initialize camera and project
    plt.close('all')
    prj = init_prj_window(*prj_screen_sz, 0.5, prj_offset)
    cam = init_cam(cam_raw_sz)

    # clear camera buffer
    for j in range(0, 100):
        _, im_cam = cam.read()

    num_im = im_prj.shape[-1]
    # num_im = 20 # for debug

    # project-and-capture, then save images
    for i in tqdm(range(num_im)):
        prj.set_data(im_prj[..., i])
        plt.pause(delay_time)  # adjust this according to your hardware and program latency
        plt.draw()

        # fetch frames from the camera buffer and drop [num_frame_delay] frames due to delay
        for j in range(0, delay_frames):
            _, im_cam = cam.read()

        # apply center crop and resize to the camera frames, then save to files
        cv.imwrite(join(cam_cap_path, 'img_{:04d}.png'.format(i + 1)), cv.resize(cc(im_cam, cam_crop_sz), cam_im_sz, interpolation=cv.INTER_AREA))
        # cv.imwrite(join(cam_cap_path, 'img_{:04d}.png'.format(i + 1)), cv.resize(im_cam, cam_im_sz, interpolation=cv.INTER_AREA))

    # release camera and close projector windows
    cam.release()
    plt.close('all')


# ------------------------------------------- Configs and logs  -----------------------------------------------
def print_sys_info():
    """
    print system information
    """
    print('-------------------------------------- System info -----------------------------------')

    # system
    print('OS: ', platform.platform())  # system build

    # pytorch and cuda
    print("torch version=" + torch.__version__)  # PyTorch version
    print("CUDA version=" + torch.version.cuda)  # Corresponding CUDA version
    # print("CUDNN version=" + torch.backends.cudnn.version())  # Corresponding cuDNN version

    # GPU count
    if torch.cuda.device_count() >= 1:
        print('Train with', torch.cuda.device_count(), 'GPUs!')
    else:
        print('Train with CPU!')

    # GPU mdoel name
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i:d} name: " + torch.cuda.get_device_name(i))  # GPU name

    print('-------------------------------------- System info -----------------------------------')


def export_setup_info(setup_path, cfg):
    """
    Export the setup info to a yml file without using OmegaConf. All sz are in (w, h) format
    :param setup_path: the setup path that contains the captured data and results
    :param prj_screen_sz: projector screen resolution
    :param cam_raw_sz: camera output frame size
    :param cam_crop_sz: a size that will be used to center crop camera output frame size, cam_crop_sz <= cam_raw_sz
    :param cam_im_sz: a size that will be used to resize center cropped camera output frame, cam_im_sz <= cam_crop_sz, and should keep aspect ratio
    :param delay_frames: how many frames to drop before we capture the correct one, increase it when ProCams are not in sync
    :param delay_time: a delay time between the project and capture operations for software sync in seconds, increase it when ProCams are not in sync
    """

    with open(join(setup_path, 'setup_info.yml'), 'w') as f:
        # f.write(setup_params)
        data = yaml.dump(cfg, f)


# generate training title string
def opt_to_string(opt):
    return f'{opt["setup_name"]}_{opt["model_name"]}_{opt["loss"]}_{opt["num_train"]}_{opt["batch_size"]}_{opt["max_iters"]}_{opt["lr"]}_{opt["lr_drop_ratio"]}_{opt["lr_drop_rate"]}_{opt["l2_reg"]}'


def init_log_file(log_dir):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_datetime = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    log_txt_filename = join(log_dir, log_datetime + '.txt')
    log_xls_filename = join(log_dir, log_datetime + '.xlsx')
    ret = pd.DataFrame(columns=['Setup', 'Model', 'Loss', 'Num train', 'Batch', 'Iters', 'PSNR', 'RMSE', 'SSIM', 'L2', 'L-inf', 'dE'])
    return ret, log_txt_filename, log_xls_filename


def write_log_file(df, log_txt_filename=None, log_xls_filename=None, mode='w'):
    if log_txt_filename is not None:
        df.to_csv(log_txt_filename, mode=mode, index=False, float_format='%.4f')
        # df.to_csv(log_txt_filename, mode=mode, index=False, float_format='%.4f', header=not os.path.exists(log_txt_filename))
    if log_xls_filename is not None:
        df.to_excel(log_xls_filename, index=False, float_format='%.4f')  # can only overwrite
        # df.to_excel(log_xls_filename, index=False, float_format='%.4f', header=not os.path.exists(log_xls_filename))  # can only overwrite
    print(f'Log files are saved to {log_txt_filename} and {log_xls_filename}')


def log_to_table(filename):
    """
    convert the old CompenNet/CompenNet++/CompenNeSt++/DeProCams log files to pandas df
    :param filename: full path to the log file
    :return: a pandas DataFrame
    """
    table = pd.read_table(filename, delim_whitespace=True)
    return table


def read_log(filename):
    """
    Read the new log txt to table
    :param filename: full path to the log file
    :return: a pandas DataFrame
    """
    table = pd.read_table(filename, sep=',')
    return table


def save_checkpoint(checkpoint_dir, model, title):
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    checkpoint_filename = abspath(join(checkpoint_dir, title + '.pth'))
    torch.save(model.state_dict(), checkpoint_filename)
    print(f'Checkpoint saved to {checkpoint_filename}\n')


def make_setup_subdirs(setup_path):
    # create basic subdirectories for the setup, some other subdirs are created later
    # prj/raw
    for sub_dir_name in ['ref', 'cb']:
        sub_dir = join(setup_path, 'prj/raw', sub_dir_name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    # cam/raw
    for sub_dir_name in ['ref', 'train', 'test', 'cb']:
        sub_dir = join(setup_path, 'cam/raw', sub_dir_name)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    # result
    ret_dir = join(setup_path, 'ret')
    if not os.path.exists(ret_dir):
        os.makedirs(ret_dir)


def idx_to_label(imgnet_labels, idx):
    # get ImageNet labels given a list/array of index
    return [list(imgnet_labels.values())[x] for x in idx]


