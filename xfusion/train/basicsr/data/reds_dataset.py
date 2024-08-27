import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from xfusion.train.basicsr.data.transforms import augment
from xfusion.train.basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from xfusion.train.basicsr.utils.flow_util import dequantize_flow
from xfusion.train.basicsr.utils.registry import DATASET_REGISTRY

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
@DATASET_REGISTRY.register()
class REDSDatasetSTF(data.Dataset):
    def __init__(self, opt):
        super(REDSDatasetSTF, self).__init__()
        self.opt = opt
        if 'num_frame_hi' not in list(opt.keys()):
            self.opt['num_frame_hi'] = 1
        
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
        while (start_frame_idx < 0) or (end_frame_idx > (self.case_frame_num[clip_name]-1)):
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

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

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
            img_results, img_flows = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], img_flows, renorm = self.opt['use_local_renorm'], saturation_thresh_percent=self.opt['saturation_percentile'], poisson_b0_exponent=self.opt['poisson_noise_b0_exp'])
        else:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], renorm = self.opt['use_local_renorm'], saturation_thresh_percent=self.opt['saturation_percentile'], poisson_b0_exponent=self.opt['poisson_noise_b0_exp'])

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

'''
@DATASET_REGISTRY.register()
class REDSDatasetSTF(data.Dataset):
    def __init__(self, opt):
        super(REDSDatasetSTF, self).__init__()
        self.opt = opt
        if 'num_frame_hi' not in list(opt.keys()):
            self.opt['num_frame_hi'] = 1
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.flow_root = Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.flag = opt['flag']
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
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
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
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
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag = self.flag, float32=True)

        # get the past high resolution frame
        img_hqs = []
        if self.opt['num_frame_hi'] == 1:
            offsets_hi = [-1]
        elif self.opt['num_frame_hi'] == 2:
            offsets_hi = [-1,1]
        for offset_hi in offsets_hi:
            if self.is_lmdb:
                img_hq_path = f'{clip_name}/{(center_frame_idx+offset_hi):08d}'
            else:
                img_hq_path = self.gt_root / clip_name / f'{(center_frame_idx+offset_hi):08d}.png'
            img_bytes = self.file_client.get(img_hq_path, 'gt')
            img_hqs.append(imfrombytes(img_bytes, flag = self.flag, float32=True))

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag = self.flag, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs, img_hqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path, img_hqs = img_hqs)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.num_frame:]

        # augmentation - flip, rotate
        if self.opt['num_frame_hi'] == 1:
            img_lqs.append(img_hqs)
        elif self.opt['num_frame_hi'] > 1:
            img_lqs.extend(img_hqs)
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-(1+self.opt['num_frame_hi'])], dim=0)
        img_gt = img_results[-1]
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
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key, 'hq': img_hqs}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key, 'hq': img_hqs}

    def __len__(self):
        return len(self.keys)
'''
@DATASET_REGISTRY.register()
class REDSDataset(data.Dataset):
    """REDS dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.flow_root = Path(opt['dataroot_flow']) if opt['dataroot_flow'] is not None else None
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.flag = opt['flag']
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
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
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 100 frames starting from 0 to 99
        while (start_frame_idx < 0) or (end_frame_idx > 99):
            center_frame_idx = random.randint(0, 99)
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
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, flag = self.flag, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag = self.flag, float32=True)
            img_lqs.append(img_lq)

        # get flows
        if self.flow_root is not None:
            img_flows = []
            # read previous flows
            for i in range(self.num_half_frames, 0, -1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_p{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_p{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)
            # read next flows
            for i in range(1, self.num_half_frames + 1):
                if self.is_lmdb:
                    flow_path = f'{clip_name}/{frame_name}_n{i}'
                else:
                    flow_path = (self.flow_root / clip_name / f'{frame_name}_n{i}.png')
                img_bytes = self.file_client.get(flow_path, 'flow')
                cat_flow = imfrombytes(img_bytes, flag='grayscale', float32=False)  # uint8, [0, 255]
                dx, dy = np.split(cat_flow, 2, axis=0)
                flow = dequantize_flow(dx, dy, max_val=20, denorm=False)  # we use max_val 20 here.
                img_flows.append(flow)

            # for random crop, here, img_flows and img_lqs have the same
            # spatial size
            img_lqs.extend(img_flows)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)
        if self.flow_root is not None:
            img_lqs, img_flows = img_lqs[:self.num_frame], img_lqs[self.num_frame:]

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        if self.flow_root is not None:
            img_results, img_flows = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'], img_flows)
        else:
            img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

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
            return {'lq': img_lqs, 'flow': img_flows, 'gt': img_gt, 'key': key}
        else:
            return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class REDSRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
