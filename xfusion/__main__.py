#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2024. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module containing an example on how to use GDAuth to access a Globus server, create 
directories and share them with users.

"""
import os
import sys
import pathlib
import argparse

from xfusion import log
from xfusion import config
from xfusion import utils
from xfusion.train import train_reds_gray
from xfusion.inference import infer

from os import path as osp
import urllib3
import shutil
import zipfile


def init(args):

    args.home.mkdir(exist_ok=True, parents=True)
    args.train_home.mkdir(exist_ok=True, parents=True)
    args.inference_home.mkdir(exist_ok=True, parents=True)
    args.log_home.mkdir(exist_ok=True, parents=True)

    if not os.path.exists(str(args.config)):
        config.write(str(args.config))
    else:
        raise RuntimeError("{0} already exists".format(args.config))


def convert(args):
    utils.compile_dataset(args)


def train(args):    
    train_reds_gray.train_pipeline(args)

def inference(args):    
    infer.inference_pipeline(args)

def download(args):
    http = urllib3.PoolManager()
    url = args.dir_inf
    path = args.out_dir_inf
    zip_file_path = pathlib.Path(path) / url.split('/')[-1]
    #pathlib.Path(path).mkdir(exist_ok=True,parents=True)
    
    with http.request('GET', url, preload_content=False) as r, open(zip_file_path, 'wb') as out_file:       
        shutil.copyfileobj(r, out_file)
    
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(path)

def main():

    # This is just to print nice logger messages
    log.setup_custom_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])

    home_params = config.HOME_PARAMS
    convert_params   = config.CONVERT_PARAMS
    train_params     = config.TRAIN_PARAMS
    inference_params = config.INFERENCE_PARAMS
    download_params = config.DOWNLOAD_PARAMS

    cmd_parsers = [
        ('init',       init,       home_params,      "Create configuration file and home directory"),
        ('convert',    convert,    convert_params,   "Convert training images to gray scale"),
        ('train',      train,      train_params,     "Train"),
        ('download',   download,   download_params,  "Download"),
        ('inference',  inference,  inference_params, "Inference"),
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)

    args = config.parse_known_args(parser, subparser=True)
    
    try:
        # load args from default (config.py) if not changed
        args._func(args)
        config.log_values(args)
        # undate globus.config file
        sections = config.XFUSION_PARAMS
        config.write(args.config, args=args, sections=sections)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
