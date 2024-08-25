import glob
import torch
from os import path as osp
from torch.utils import data as data
from basicsr.data.data_util import duf_downsample, read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY

import random
import numpy as np

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

'''
def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None, img_hqs = None):
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
    top = random.randint(0, h_lq - lq_patch_size[0])
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
'''
@DATASET_REGISTRY.register()
class VideoTestDatasetSTF(data.Dataset):
    def __init__(self, opt):
        super(VideoTestDatasetSTF, self).__init__()
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
            
            with open(opt['meta_info_file'], 'r') as fin:
                lines = fin.readlines()
            subfolders = [line.split(' ')[0] for line in lines]
            subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
            subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
            for line in lines:
                self.case_frame_size[line.split(' ')[0]] = tuple(map(int, line.split(' ')[-1].split('\n')[0][1:-1].split(',')))
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial']:
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
            if 'use_local_renorm' in self.opt.keys():
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

'''
@DATASET_REGISTRY.register()
class VideoTestDatasetSTF(data.Dataset):
    def __init__(self, opt):
        super(VideoTestDatasetSTF, self).__init__()
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
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial']:
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
        if ('scale' in self.opt.keys()) and ('gt_size' in self.opt.keys()):
            scale = self.opt['scale']
            gt_size = self.opt['gt_size']

        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])
        if self.opt['num_frame_hi'] == 2:
            select_idx_hi = [select_idx[int((len(select_idx)+1)/2-2)], select_idx[int((len(select_idx)+1)/2)]]
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
            img_gt.squeeze_(0)
            img_hq.squeeze_(0)

        if ('scale' in self.opt.keys()) and ('gt_size' in self.opt.keys()):
            img_gt, imgs_lq, img_hq = paired_random_crop(img_gt.unsqueeze(0), imgs_lq, gt_size, scale, img_hqs = img_hq)
            img_gt = img_gt.squeeze(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'hq': img_hq,
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])
'''

@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    ::

        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder2
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
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
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial']:
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
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq,flag=self.flag)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]],flag=self.flag)
            img_gt.squeeze_(0)

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


@DATASET_REGISTRY.register()
class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        io_backend (dict): IO backend type and other kwarg.
        cache_data (bool): Whether to cache testing datasets.
        name (str): Dataset name.
        meta_info_file (str): The path to the file storing the list of test folders. If not provided, all the folders
            in the dataroot will be used.
        num_frame (int): Window size for input frames.
        padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = read_img_seq(lq_path)
        img_gt = read_img_seq([gt_path])
        img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': self.data_info['folder'][index],  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/843
            'border': self.data_info['border'][index],  # 0 for non-border
            'lq_path': lq_path[self.opt['num_frame'] // 2]  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestDUFDataset(VideoTestDataset):
    """ Video test dataset for DUF dataset.

    Args:
        opt (dict): Config for train dataset. Most of keys are the same as VideoTestDataset.
            It has the following extra keys:
        use_duf_downsampling (bool): Whether to use duf downsampling to generate low-resolution frames.
        scale (bool): Scale, which will be added automatically.
    """

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            if self.opt['use_duf_downsampling']:
                # read imgs_gt to generate low-resolution frames
                imgs_lq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            if self.opt['use_duf_downsampling']:
                img_paths_lq = [self.imgs_gt[folder][i] for i in select_idx]
                # read imgs_gt to generate low-resolution frames
                imgs_lq = read_img_seq(img_paths_lq, require_mod_crop=True, scale=self.opt['scale'])
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
                imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]], require_mod_crop=True, scale=self.opt['scale'])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }


@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        opt (dict): Same as VideoTestDataset. Unused opt:
        padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            raise NotImplementedError('Without cache_data is not implemented.')

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
