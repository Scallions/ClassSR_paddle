import os
import sys
import time
import math
import paddle.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import paddle
from shutil import get_terminal_size

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



def cal_FLOPs(which_model,num_ress):
    if which_model =='classSR_3class_fsrcnn_net':
        flops = ((141 * num_ress[0] + 304 * num_ress[1] + 468 * num_ress[2]) / sum(num_ress))+7.4
        percent=flops/468
    elif which_model=='classSR_3class_carn':
        flops = (0.38 * num_ress[0] + 0.77 * num_ress[1] + 1.15 * num_ress[2]) / sum(num_ress)
        percent = flops / 1.15
    elif which_model=='classSR_3class_srresnet':
        flops = (1.66 * num_ress[0] + 3.44 * num_ress[1] + 5.20 * num_ress[2]) / sum(num_ress)
        percent = flops / 5.20
    elif which_model=='classSR_3class_rcan':
        flops = (10.33 * num_ress[0] + 19.90 * num_ress[1] + 32.60 * num_ress[2]) / sum(num_ress)
        percent = flops / 32.60
    return flops,percent




def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    #torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (paddle.is_tensor(tensor) or
            (isinstance(tensor, list) and all(paddle.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
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
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = paddle.slice(grid,[1],y * height + padding, height - padding)
            grid = paddle.slice(grid,[2],x * width + padding, width - padding).copy_(tensor[k])
            #grid.narrow(1, y * height + padding, height - padding)\
            #    .narrow(2, x * width + padding, width - padding)\
            #    .copy_(tensor[k])
            k = k + 1
    return grid


# TODO: torch make grid
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = paddle.to_tensor(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with paddle.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, paddle.flip(inp, (-1, )))
    output_f = output_f + paddle.flip(output, (-1, ))
    # flip H
    output = single_forward(model, paddle.flip(inp, (-2, )))
    output_f = output_f + paddle.flip(output, (-2, ))
    # flip both H and W
    output = single_forward(model, paddle.flip(inp, (-2, -1)))
    output_f = output_f + paddle.flip(output, (-2, -1))

    return output_f / 4


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float(80)
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
