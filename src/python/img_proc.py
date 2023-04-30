import sys
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import copy

import torchvision.transforms
from PIL import ImageFont, ImageDraw, ImageOps
from skimage.filters import threshold_multiotsu


def threshold_im(im_in, compensation=False):
    # find the direct light binary mask for SPAA
    if im_in.ndim == 3:
        # get rid of out of range values
        im_in = np.clip(im_in, 0, 1)

        im_in = cv.cvtColor(im_in, cv.COLOR_RGB2GRAY)  # !!very important, result of COLOR_RGB2GRAY is different from COLOR_BGR2GRAY
        if im_in.dtype == 'float32':
            im_in = np.uint8(im_in * 255)
        if (compensation):
            # _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
            levels = 4
            thresholds = threshold_multiotsu(cv.GaussianBlur(im_in, (3, 3), 1.5), levels)

            # Quantized image
            im_mask = np.digitize(im_in, bins=thresholds)
            im_mask = im_mask > 2
        else:
            levels = 2
            im_in_smooth = cv.GaussianBlur(im_in, (3, 3), 1.5)
            thresholds = threshold_multiotsu(im_in_smooth, levels)

            # # Quantized image
            im_mask = np.digitize(im_in_smooth, bins=thresholds)
            im_mask = im_mask > 0

    elif im_in.dtype == np.bool:  # if already a binary image
        im_mask = im_in

    # find the largest contour by area then convert it to convex hull
    contours, hierarchy = cv.findContours(np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if (compensation):
        max_contours = max(contours, key=cv.contourArea)
        hulls = cv.convexHull(max(contours, key=cv.contourArea))
    else:
        max_contours = np.concatenate(contours)  # instead of use the largest area, we use them all
        hulls = cv.convexHull(max_contours)
    im_roi = cv.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0

    # also calculate the bounding box
    # bbox = cv.boundingRect(max(contours, key=cv.contourArea))
    bbox = cv.boundingRect(max_contours)
    corners = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]

    # normalize to (-1, 1) following pytorch grid_sample coordinate system
    h = im_in.shape[0]
    w = im_in.shape[1]

    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1

    return im_mask, im_roi, corners


def checkerboard(*args):
    # Python implementation of MATLAB's checkerboard function
    # Parse inputs
    n = 10
    p = 4
    q = p

    if len(args) == 1:
        n = args[0]
    elif len(args) == 2:
        n, p = args
        q = p
    elif len(args) == 3:
        n, p, q = args

    # Generate tile
    tile = np.tile(np.kron([[0, 1], [1, 0]], np.ones((n, n))), (1, 1))

    # Create checkerboard
    if q % 2 == 0:
        # Make left and right sections separately
        num_col_reps = int(np.ceil(q / 2))
        ileft = np.tile(tile, (p, num_col_reps))

        tile_right = np.tile(np.kron([[0, 0.7], [0.7, 0]], np.ones((n, n))), (1, 1))
        iright = np.tile(tile_right, (p, num_col_reps))

        # Tile the left and right halves together
        checkerboard = np.concatenate((ileft, iright), axis=1)
    else:
        # Make the entire image in one shot
        checkerboard = np.tile(tile, (p, q))

        # Make right half plane have light gray tiles
        mid_col = int(checkerboard.shape[1] / 2) + 1
        checkerboard[:, mid_col:] = checkerboard[:, mid_col:] - .3
        checkerboard[np.where(checkerboard < 0)] = 0

    return checkerboard.astype('float64')


# ------------------------------------------- Tensor operations  -----------------------------------------------
def expand_4d(x):
    # expand a 1D,2D,3D tensor to 4D tensor (BxCxHxW)
    for i in range(4 - x.ndim):
        x = x[None]
    return x

# resize a tensor using F.interpolate (note we need to convert it to 4D tensor (BxCxHxW) in order to use 2D size, e.g., (224, 224)
def resize(x, size):
    if x.ndim == 2:
        return F.interpolate(x[None, None], size, mode='area')[0][0]
    if x.ndim == 3:
        return F.interpolate(x[None], size, mode='area')[0]
    if x.ndim == 4:
        return F.interpolate(x, size, mode='area')


def center_crop(x, size):
    # center crop an image by size
    h, w = x.shape[-2:]
    th, tw = size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return x[..., i:i + th, j:j + tw]


def create_gray_pattern(w, h):
    # Python implementation of MATLAB's createGrayPattern
    nbits = np.ceil(np.log2([w, h])).astype(int)  # # of bits for vertical/horizontal patterns
    offset = (2 ** nbits - [w, h]) // 2  # offset the binary pattern to be symmetric

    # coordinates to binary code
    c, r = np.meshgrid(np.arange(w), np.arange(h))
    bin_pattern = [np.unpackbits((c + offset[0])[..., None].view(np.uint8), axis=-1, bitorder='little', count=nbits[0])[..., ::-1],
        np.unpackbits((r + offset[1])[..., None].view(np.uint8), axis=-1, bitorder='little', count=nbits[1])[..., ::-1]]

    # binary pattern to gray pattern
    gray_pattern = copy.deepcopy(bin_pattern)
    for n in range(len(bin_pattern)):
        for i in range(1, bin_pattern[n].shape[-1]):
            gray_pattern[n][:, :, i] = np.bitwise_xor(bin_pattern[n][:, :, i - 1], bin_pattern[n][:, :, i])

    # allPatterns also contains complementary patterns and all 0/1 patterns
    prj_patterns = np.zeros((h, w, 2 * sum(nbits) + 2), dtype=np.uint8)
    prj_patterns[:, :, 0] = 1  # All ones pattern

    # Vertical
    for i in range(gray_pattern[0].shape[-1]):
        prj_patterns[:, :, 2 * i + 2] = gray_pattern[0][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 3] = np.logical_not(gray_pattern[0][:, :, i]).astype(np.uint8)

    # Horizontal
    for i in range(gray_pattern[1].shape[-1]):
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 2] = gray_pattern[1][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 3] = np.logical_not(gray_pattern[1][:, :, i]).astype(np.uint8)

    prj_patterns *= 255

    # to RGB image
    # prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (0, 1, 3, 2))  # to (h, w, c, n)
    prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (2, 0, 1, 3))  # to (n, h, w, c)

    return prj_patterns


def insert_text(x, text, pos, font_sz=14, color=(0, 0, 0)):
    # insert text to a 3D tensor (CxHxW)
    im = torchvision.transforms.ToPILImage()(x.detach().cpu())

    # font
    if sys.platform == 'win32':
        font = ImageFont.truetype("arial.ttf", font_sz)
    elif sys.platform == 'linux':
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_sz)
    else:
        font = ImageFont.load_default()

    # insert text
    draw = ImageDraw.Draw(im)
    draw.text(pos, text, color, font=font)

    return torchvision.transforms.ToTensor()(im)


def expand_boarder(x, border=(0, 20, 0, 0), fill=(255, 255, 255)):
    # expand a tensor image's boarder
    im = torchvision.transforms.ToPILImage()(x.detach().cpu())
    im = ImageOps.expand(im, border=border, fill=fill)
    return torchvision.transforms.ToTensor()(im)


def to_pseudocolor(x, colormap=cv.COLORMAP_BONE):
    # convert a tensor x, shape=[1, H, W, 1] to pseudo-color image for visualization, e.g., 1-channel depth map
    device = x.device
    x = x.squeeze()
    x = (x - x.min()) / (x.max() - x.min())
    x = cv.applyColorMap(np.uint8(x.cpu().numpy() * 255), colormap)
    x = (torch.Tensor(cv.cvtColor(x, cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255).to(device)
    return x

