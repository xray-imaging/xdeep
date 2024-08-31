import os
import yaml
import numpy as np

from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from collections import OrderedDict
from PIL import Image

from xfusion import log


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

def compile_dataset(args):
    cases_hi = natsorted(list(args.dir_hi.glob('*')))
    for case_hi in tqdm(cases_hi):
        files_hi = natsorted(list(case_hi.glob('*.png')))
        out_case_hi = args.out_dir_hi / case_hi.stem
        Path(out_case_hi).mkdir(exist_ok=True,parents=True)
        out_case_lo = args.out_dir_lo / case_hi.stem
        Path(out_case_lo).mkdir(exist_ok=True,parents=True)
        for file_hi in files_hi:
            img_hi  = Image.open(file_hi).convert('L')
            file_lo = args.dir_lo / case_hi.stem / file_hi.name
            img_lo  = Image.open(file_lo).convert('L')
            img_hi  = np.array(img_hi)
            img_lo  = np.array(img_lo)
            img_hi  = Image.fromarray(np.concatenate([img_hi[:,:,None],img_hi[:,:,None],img_hi[:,:,None]],axis=2))
            img_lo  = Image.fromarray(np.concatenate([img_lo[:,:,None],img_lo[:,:,None],img_lo[:,:,None]],axis=2))
            log.info("Converting to gray scale low (%s) and high (%s) res images" % (file_lo, file_hi))
            img_hi.save(out_case_hi / file_hi.name)
            img_lo.save(out_case_lo / file_hi.name)


