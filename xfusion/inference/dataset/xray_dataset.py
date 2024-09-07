import cv2
import itertools
import random
import torch
import numpy as np
from os import path as osp
import torchvision.transforms.functional as TF
from torch.utils import data as data
import os, glob
from pathlib import Path

#import sys
#sys.path.insert(0,'/home/beams/FAST/conda/BasicSR_single_channel')
from xfusion.inference.dataset.file_client import FileClient
from xfusion.inference.dataset.logger import get_root_logger

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if flag == 'grayscale':
        img = img.reshape((img.shape[0],img.shape[1],-1))
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False, flag = 'color'):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]
    if flag == 'grayscale':
        imgs = [img[:,:,0,None] for img in imgs]
    elif flag == 'color':
        pass
    else:
        raise Exception
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False, renorm=False, saturation_thresh_percent = 0,\
            poisson_b0_exponent = False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]

    if renorm:
        assert (saturation_thresh_percent >=0) and (saturation_thresh_percent < 50)
        img_min_max = np.quantile(np.concatenate([img.flatten() for img in imgs]),np.array([saturation_thresh_percent/100, 1-saturation_thresh_percent/100]))
        img_min = img_min_max[0]#min([img.min() for img in imgs])
        img_max = img_min_max[1]#max([img.max() for img in imgs])
        if img_min == img_max:
            pass
        else:
            imgs = [np.clip(img, img_min, img_max) for img in imgs]
            imgs = [(img-img_min)/(img_max-img_min) for img in imgs]
    imgs = [_augment(img) for img in imgs]
    if poisson_b0_exponent:
        assert len(poisson_b0_exponent) == 2
        assert poisson_b0_exponent[0] <= poisson_b0_exponent[1]
        img_sz = [img.size for img in imgs]
        b0 = 10**(np.random.uniform(poisson_b0_exponent[0], poisson_b0_exponent[1]))
        imgs = [np.clip(np.random.poisson(img * b0) / b0,0,1) if img_sz[idx] == min(img_sz) else img for idx, img in enumerate(imgs)]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None, img_hqs = None, patch_corner = None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """
    if not isinstance(gt_patch_size,list):
        gt_patch_size = [gt_patch_size,gt_patch_size]
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if img_hqs is not None:
        if not isinstance(img_hqs, list):
            img_hqs = [img_hqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
        if img_hqs is not None:
            h_hq, w_hq = img_hqs[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
        if img_hqs is not None:
            h_hq, w_hq = img_hqs[0].shape[0:2]

    lq_patch_size = [ps // scale for ps in gt_patch_size]

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size[0] or w_lq < lq_patch_size[1]:
        pass
        #print(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
        #                 f'({lq_patch_size[0]}, {lq_patch_size[1]}). '
        #                 f'Please remove {gt_path}.')
    if img_hqs is not None:
        if h_gt != h_hq or w_gt != w_hq:
            raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not same as ',
                            f'HQ ({h_hq}, {w_hq}).')

    if h_lq < lq_patch_size[0]:
        top_pad = lq_patch_size[0] - h_lq
        if input_type == 'Tensor':
            img_lqs = [torch.nn.functional.pad(v, (0,0,0,top_pad),'reflect') for v in img_lqs]
            h_lq, w_lq = img_lqs[0].size()[-2:]
        else:
            pad_size = tuple([(0,top_pad) if d == 0 else (0,0) for d in range(img_lqs[0].ndim)])
            img_lqs = [np.pad(v,pad_size,mode='reflect') for v in img_lqs]
            h_lq, w_lq = img_lqs[0].shape[0:2]

        top_pad_gt = int(top_pad * scale)
        if input_type == 'Tensor':
            img_gts = [torch.nn.functional.pad(v,(0,0,0,top_pad_gt),'reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            pad_size = tuple([(0,top_pad_gt) if d == 0 else (0,0) for d in range(img_gts[0].ndim)])
            img_gts = [np.pad(v,pad_size,mode='reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].shape[0:2]

        if img_hqs is not None:
            if input_type == 'Tensor':
                img_hqs = [torch.nn.functional.pad(v,(0,0,0,top_pad_gt),'reflect') for v in img_hqs]
                h_hq, w_hq = img_hqs[0].size()[-2:]
            else:
                pad_size = tuple([(0,top_pad_gt) if d == 0 else (0,0) for d in range(img_hqs[0].ndim)])
                img_hqs = [np.pad(v,pad_size,mode='reflect') for v in img_hqs]
                h_hq, w_hq = img_hqs[0].shape[0:2]


    if w_lq < lq_patch_size[1]:
        left_pad = lq_patch_size[1] - w_lq
        if input_type == 'Tensor':
            img_lqs = [torch.nn.functional.pad(v, (0,left_pad,0,0),'reflect') for v in img_lqs]
            h_lq, w_lq = img_lqs[0].size()[-2:]
        else:
            pad_size = tuple([(0,left_pad) if d == 1 else (0,0) for d in range(img_lqs[0].ndim)])
            img_lqs = [np.pad(v,pad_size,mode='reflect') for v in img_lqs]
            h_lq, w_lq = img_lqs[0].shape[0:2]

        left_pad_gt = int(left_pad * scale)
        if input_type == 'Tensor':
            img_gts = [torch.nn.functional.pad(v,(0,left_pad_gt,0,0),'reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].size()[-2:]
        else:
            pad_size = tuple([(0,left_pad_gt) if d == 1 else (0,0) for d in range(img_gts[0].ndim)])
            img_gts = [np.pad(v,pad_size,mode='reflect') for v in img_gts]
            h_gt, w_gt = img_gts[0].shape[0:2]

        if img_hqs is not None:
            if input_type == 'Tensor':
                img_hqs = [torch.nn.functional.pad(v,(0,left_pad_gt,0,0),'reflect') for v in img_hqs]
                h_hq, w_hq = img_hqs[0].size()[-2:]
            else:
                pad_size = tuple([(0,left_pad_gt) if d == 1 else (0,0) for d in range(img_hqs[0].ndim)])
                img_hqs = [np.pad(v,pad_size,mode='reflect') for v in img_hqs]
                h_hq, w_hq = img_hqs[0].shape[0:2]

    # randomly choose top and left coordinates for lq patch
    if patch_corner is None:
        top = random.randint(0, h_lq - lq_patch_size[0])
        left = random.randint(0, w_lq - lq_patch_size[1])
    else:
        assert len(patch_corner) == 2
        top = patch_corner[0]
        left = patch_corner[1]
        if (top < 0) | (top > (h_lq - lq_patch_size[0])):
            top = random.randint(0, h_lq - lq_patch_size[0])
        if (left < 0) | (left > (w_lq - lq_patch_size[1])):
            left = random.randint(0, w_lq - lq_patch_size[1])
    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size[0], left:left + lq_patch_size[1]] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size[0], left:left + lq_patch_size[1], ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1]] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1], ...] for v in img_gts]

    if img_hqs is not None:
        if input_type == 'Tensor':
            img_hqs = [v[:, :, top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1]] for v in img_hqs]
        else:
            img_hqs = [v[top_gt:top_gt + gt_patch_size[0], left_gt:left_gt + gt_patch_size[1], ...] for v in img_hqs]


    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if img_hqs is not None:
        if len(img_hqs) == 1:
            img_hqs = img_hqs[0]
    if img_hqs is None:
        return img_gts, img_lqs
    else:
        return img_gts, img_lqs, img_hqs

def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection', mult_spacing = 1):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad * mult_spacing, crt_idx + num_pad * mult_spacing + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices

class XrayDatasetSTF(data.Dataset):
    def __init__(self, opt):
        super(XrayDatasetSTF, self).__init__()
        self.opt = opt
        if 'num_frame_hi' not in list(opt.keys()):
            self.opt['num_frame_hi'] = 1

        if opt['name'].lower() in ['redsctc','xray']:
            assert type(opt['dataroot_gt']) is list
            self.gt_root, self.lq_root = [Path(gr) for gr in opt['dataroot_gt']], [Path(lr) for lr in opt['dataroot_lq']]
        else:
            if type(opt['dataroot_gt']) is str:
                self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
            else:
                raise Exception('data type not recognized...')
        self.flow_root = Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.flag = opt['flag']
        self.keys = []

        self.case_frame_num = {}
        self.case_gt_root = {}
        self.case_lq_root = {}
        if opt['name'].lower() in ['redsctc','xray']:
            assert type(opt['dataroot_gt']) is list
            assert type(opt['dataroot_lq']) is list
            assert type(opt['meta_info_file']) is list

            for k, meta_info in enumerate(opt['meta_info_file']):
                with open(meta_info, 'r') as fin:
                    for line in fin:
                        folder, frame_num, _ = line.split(' ')
                        self.case_frame_num[folder] = int(frame_num)
                        self.case_gt_root[folder] = self.gt_root[k]
                        self.case_lq_root[folder] = self.lq_root[k]
                        self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])
        else:
            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.case_frame_num[folder] = int(frame_num)
                    self.case_gt_root[folder] = self.gt_root
                    self.case_lq_root[folder] = self.lq_root
                    self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        elif opt['val_partition'] == 'officialCTC':
            val_partition = [f'{v:03d}' for v in range(240, 270)] + ['282', '283', '284', '285']
        elif opt['val_partition'] == 'addma':
            val_partition = [f'{v:03d}' for v in [436,536,636,736]]
        elif opt['val_partition'] == 'insect':
            val_partition = [f'{v:03d}' for v in [836]]
        elif opt['val_partition'] == 'insect_addma':
            val_partition = [f'{v:03d}' for v in [436,536,636,736,836]]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4', 'REDS4CTC,'addma','insect','insect_addma'].")
        self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        if 'hi_interval_mult' in list(opt.keys()):
            self.hi_interval_mult = opt['hi_interval_mult']
        else:
            self.hi_interval_mult = 1
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
        if 'sample_patch_ok' in list(opt.keys()):
            self.sample_patch_ok = opt['sample_patch_ok']
        else:
            self.sample_patch_ok = False

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list) if int(clip_name) >= 270 else 1

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        # remove first frame of Shimadzu data
        while ((int(clip_name) < 2234) and ((start_frame_idx < 0) or (end_frame_idx > (self.case_frame_num[clip_name]-1)))) or\
              ((int(clip_name) >= 2234) and ((start_frame_idx < 1) or (end_frame_idx > (self.case_frame_num[clip_name]-1)))):
            center_frame_idx = random.randint(0, self.case_frame_num[clip_name]-1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            #if type(self.gt_root) is list:
            #    img_gt_path = self.case_gt_root[clip_name] / clip_name / f'{frame_name}.png'
            #else:
            #    img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
            img_gt_path = self.case_gt_root[clip_name] / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag = self.flag, float32=True)

        # get the past high resolution frame
        if self.opt['num_frame_hi'] == 0:
            img_hqs = None
        else:
            img_hqs = []
            if self.opt['num_frame_hi'] == 1:
                offsets_hi = [-1 * interval]#[int(start_frame_idx-center_frame_idx)]
            elif self.opt['num_frame_hi'] == 2:
                offsets_hi = [-1 * interval,1 * interval]# [int(start_frame_idx-center_frame_idx), int(end_frame_idx-center_frame_idx)]
            for offset_hi in offsets_hi:
                offset_mult = random.randint(1,self.hi_interval_mult)
                if ((center_frame_idx + offset_hi * offset_mult) >= 0) and ((center_frame_idx + offset_hi * offset_mult) <= (self.case_frame_num[clip_name]-1)):
                    offset_hi = offset_hi * offset_mult

                if self.is_lmdb:
                    img_hq_path = f'{clip_name}/{(center_frame_idx+offset_hi):08d}'
                else:
                    #if type(self.gt_root) is list:
                    #    img_hq_path = self.case_gt_root[clip_name] / clip_name / f'{(center_frame_idx+offset_hi):08d}.png'
                    #else:
                    #    img_hq_path = self.gt_root / clip_name / f'{(center_frame_idx+offset_hi):08d}.png'
                    img_hq_path = self.case_gt_root[clip_name] / clip_name / f'{(center_frame_idx+offset_hi):08d}.png'
                img_bytes = self.file_client.get(img_hq_path, 'gt')
                img_hqs.append(imfrombytes(img_bytes, flag = self.flag, float32=True))

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
            else:
                #if type(self.lq_root) is list:
                #    img_lq_path = self.case_lq_root[clip_name] / clip_name / f'{neighbor:08d}.png'
                #else:
                #    img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_lq_path = self.case_lq_root[clip_name] / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag = self.flag, float32=True)
            img_lqs.append(img_lq)
            if self.sample_patch_ok and (clip_name in ['270', '271', '272', '273']):
                if len(img_lqs)==2:
                    img_lq_diff = (abs(img_lq - img_lqs[-2])**4).astype(np.float64)
                    #img_lq_diff = img_lq_diff - img_lq_diff.mean()
                    sample_patch_probs = (img_lq_diff / img_lq_diff.sum()).reshape((-1,1)).squeeze() #(np.exp(img_lq_diff) / np.exp(img_lq_diff).sum()).reshape((-1,1)).squeeze()
                    grid_idx = np.where(np.random.multinomial(1,sample_patch_probs/sample_patch_probs.sum(),1))[1][0]
                    grid_idx_ = []
                    img_lq_grids = np.indices(img_lq_diff.shape)
                    for d in range(len(list(img_lq_diff.shape))):
                        grid_idx_.append(img_lq_grids[d].reshape((-1,1)).squeeze()[grid_idx])
                    if grid_idx_[-1] == 0:
                        grid_idx_ = grid_idx_[:-1]
                    assert len(grid_idx_) == len(gt_size)
                    patch_corner = [grid_idx_[i] - gt_size[i] // (scale * 2) for i in range(len(grid_idx_))]
                    patch_corner = [max(0, pc) for pc in patch_corner]
                    patch_corner = [min(pc, img_lq_diff.shape[i] - gt_size[i] // scale - 1) for i, pc in enumerate(patch_corner)]
                    patch_corner = tuple(patch_corner)
            else:
                patch_corner = None

        

        # randomly crop
        if self.opt['num_frame_hi'] == 0:
            img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path, img_hqs = img_hqs, patch_corner = patch_corner)
        else:
            img_gt, img_lqs, img_hqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path, img_hqs = img_hqs, patch_corner = patch_corner)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.num_frame:]

        # augmentation - flip, rotate
        if self.opt['num_frame_hi'] == 1:
            img_lqs.append(img_hqs)
        elif self.opt['num_frame_hi'] > 1:
            img_lqs.extend(img_hqs)
        img_lqs.append(img_gt)
        if 'use_local_renorm' not in list(self.opt.keys()):
            self.opt['use_local_renorm'] = False
        if 'saturation_percentile' not in list(self.opt.keys()):
            self.opt['saturation_percentile'] = 0
        if 'poisson_noise_b0_exp' not in list(self.opt.keys()):
            self.opt['poisson_noise_b0_exp'] = False
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], img_flows, renorm = self.opt['use_local_renorm'], saturation_thresh_percent=self.opt['saturation_percentile'],poisson_b0_exponent=self.opt['poisson_noise_b0_exp'])
        else:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], renorm = self.opt['use_local_renorm'], saturation_thresh_percent=self.opt['saturation_percentile'],poisson_b0_exponent=self.opt['poisson_noise_b0_exp'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-(1+self.opt['num_frame_hi'])], dim=0)
        img_gt = img_results[-1]
        if self.opt['num_frame_hi'] == 0:
            pass
        else:
            img_hqs = torch.stack(img_results[-(1+self.opt['num_frame_hi']):-1], dim=0)

        if self.flow_root is not None:
            img_flows = img2tensor(img_flows)
            # add the zero center flow
            img_flows.insert(self.num_half_frames, torch.zeros_like(img_flows[0]))
            img_flows = torch.stack(img_flows, dim=0)

        # img_lqs: (t, c, h, w)
        # img_flows: (t, 2, h, w)
        # img_gt: (c, h, w)
        # key: str
        if self.flow_root is not None:
            if img_hqs is not None:
                return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key, 'hq': img_hqs}
            else:
                return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            if img_hqs is not None:
                return {'lq': img_lqs, 'gt': img_gt, 'key': key, 'hq': img_hqs}
            else:
                return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)

class XrayVideoTestDatasetSTF(data.Dataset):
    def __init__(self, opt):
        super(XrayVideoTestDatasetSTF, self).__init__()
        self.opt = opt
        if 'num_frame_hi' not in list(opt.keys()):
            self.opt['num_frame_hi'] = 1
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'
        self.flag = opt['flag']
        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}

        if 'meta_info_file' in opt:

            self.case_frame_size = {}
            if opt['name'].lower() == 'xray':
                assert type(opt['meta_info_file']) is list
                subfolders, subfolders_lq, subfolders_gt = [], [], []
                for k, meta_info in enumerate(opt['meta_info_file']):
                    with open(meta_info, 'r') as fin:
                        lines = fin.readlines()

                    subfolders_ = [line.split(' ')[0] for line in lines]
                    subfolders.extend(subfolders_)
                    subfolders_lq.extend([osp.join(self.lq_root[k], key) for key in subfolders_])
                    subfolders_gt.extend([osp.join(self.gt_root[k], key) for key in subfolders_])
                    for line in lines:
                        self.case_frame_size[line.split(' ')[0]] = list(map(int, line.split(' ')[-1].split('\n')[0][1:-1].split(',')))
            else:
                with open(opt['meta_info_file'], 'r') as fin:
                    lines = fin.readlines()
                subfolders = [line.split(' ')[0] for line in lines]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
                for line in lines:
                    self.case_frame_size[line.split(' ')[0]] = tuple(map(int, line.split(' ')[-1].split('\n')[0][1:-1].split(',')))
        else:
            if opt['name'].lower() =='xray':
                #subfolders_lq = [sorted(glob.glob(osp.join(lr, '*'))) for lr in self.lq_root]
                subfolders_lq = list(sorted(self.lq_root))
                #subfolders_gt = [sorted(glob.glob(osp.join(gr, '*'))) for gr in self.gt_root]
                subfolders_gt = list(sorted(self.gt_root))
            else:
                subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
                subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() == 'xray':
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                self.data_info['lq_path'].extend(img_paths_lq)
                self.data_info['gt_path'].extend(img_paths_gt)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')

    def __getitem__(self, index):

        scale = self.opt['scale']


        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]
        
        gt_size = self.opt['gt_size']

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])
        if self.opt['num_frame_hi'] == 2:
            select_idx_hi = [select_idx[int((len(select_idx)+1)/2-2)], select_idx[int((len(select_idx)+1)/2)]]#[select_idx[0],select_idx[-1]]
        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
            if self.opt['num_frame_hi'] == 1:
                img_hq = self.imgs_gt[folder][select_idx[int((len(select_idx)+1)/2-2)]]
            elif self.opt['num_frame_hi'] == 2:
                img_hq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx_hi))
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq,flag=self.flag)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]],flag=self.flag)
            if self.opt['num_frame_hi'] == 1:
                img_hq = read_img_seq([self.imgs_gt[folder][select_idx[int((len(select_idx)+1)/2-2)]]],flag=self.flag)
            elif self.opt['num_frame_hi'] == 2:
                img_paths_hq = [self.imgs_gt[folder][i] for i in select_idx_hi]
                img_hq = read_img_seq(img_paths_hq,flag=self.flag)
            #img_gt.squeeze_(0)
            #img_hq.squeeze_(0)
        if self.opt['num_frame_hi'] == 0:
            img_hq = None
            img_gt, imgs_lq = paired_random_crop(img_gt, imgs_lq, gt_size, scale, img_hqs = img_hq)
            if self.opt['use_local_renorm']:
                assert (self.opt['saturation_percentile'] >=0) and (self.opt['saturation_percentile'] < 50)
                img_min_max = torch.quantile(torch.cat([img.flatten() for img in [img_gt,imgs_lq]]),torch.tensor([self.opt['saturation_percentile']/100, 1-self.opt['saturation_percentile']/100]))
                img_min = img_min_max[0]#min([img.min() for img in imgs])
                img_max = img_min_max[1]#max([img.max() for img in imgs])
                if img_min == img_max:
                    pass
                else:
                    img_gt = torch.clamp(img_gt, img_min, img_max)
                    imgs_lq = torch.clamp(imgs_lq, img_min, img_max)
                    img_gt = (img_gt-img_min)/(img_max-img_min)
                    imgs_lq = (imgs_lq-img_min)/(img_max-img_min)

        else:
            img_gt, imgs_lq, img_hq = paired_random_crop(img_gt, imgs_lq, gt_size, scale, img_hqs = img_hq)

            if 'use_local_renorm' not in list(self.opt.keys()):
                self.opt['use_local_renorm'] = False
            if 'saturation_percentile' not in list(self.opt.keys()):
                self.opt['saturation_percentile'] = 0

            if self.opt['use_local_renorm']:
                assert (self.opt['saturation_percentile'] >=0) and (self.opt['saturation_percentile'] < 50)
                img_min_max = torch.quantile(torch.cat([img.flatten() for img in [img_gt,imgs_lq,img_hq]]),torch.tensor([self.opt['saturation_percentile']/100, 1-self.opt['saturation_percentile']/100]))
                img_min = img_min_max[0]#min([img.min() for img in imgs])
                img_max = img_min_max[1]#max([img.max() for img in imgs])
                if img_min == img_max:
                    pass
                else:
                    img_gt = torch.clamp(img_gt, img_min, img_max)
                    imgs_lq = torch.clamp(imgs_lq, img_min, img_max)
                    img_hq = torch.clamp(img_hq, img_min, img_max)
                    img_gt = (img_gt-img_min)/(img_max-img_min)
                    imgs_lq = (imgs_lq-img_min)/(img_max-img_min)
                    img_hq = (img_hq-img_min)/(img_max-img_min)

        img_gt = img_gt.squeeze(0)
        if img_hq is not None:
            return {
                'lq': imgs_lq,  # (t, c, h, w)
                'gt': img_gt,  # (c, h, w)
                'hq': img_hq,
                'folder': folder,  # folder name
                'idx': self.data_info['idx'][index],  # e.g., 0/99
                'border': border,  # 1 for border, 0 for non-border
                'lq_path': lq_path  # center frame
            }
        else:
            return {
                'lq': imgs_lq,  # (t, c, h, w)
                'gt': img_gt,  # (c, h, w)
                'folder': folder,  # folder name
                'idx': self.data_info['idx'][index],  # e.g., 0/99
                'border': border,  # 1 for border, 0 for non-border
                'lq_path': lq_path  # center frame
            }

    def __len__(self):
        return len(self.data_info['gt_path'])