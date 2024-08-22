################## 1. Download checkpoints and build models
import os
import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
import io 
import torch
import h5py
from data.dataset_utils import *
import tempfile
import pickle
from data.utils import *
from cloudpathlib import S3Path
import boto3
from boto3.s3.transfer import TransferConfig
import botocore
from concurrent.futures import ThreadPoolExecutor
import backoff
import torch
#from torch.utils.data import DataLoader
from wavelet_datamodule import WaveletDataModule
import dist
from utils import arg_util, misc
#from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
import argparse
import json
import yaml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from accelerate import Accelerator
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
args: arg_util.Args = arg_util.init_dist_and_get_args()
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


@backoff.on_exception(backoff.expo, botocore.exceptions.ClientError, max_time=420)
def get_file_s3():
    try:
        s3_client = get_s3_client('s3')
        bucket = 'vqvae-training-output'
        s3_file_path = 'Wavelet/VQVAE-STAGE1-Wavelet _20240801-022632/TorchTrainer_2c5a0_00000_0_2024-08-01_02-26-46/checkpoint_000299/checkpoint.ckpt'
        s3_response_object = s3_client.get_object(Bucket=bucket, Key=s3_file_path)
        object_content = s3_response_object['Body'].read()
    except botocore.exceptions.ClientError:
        print(f"Error downloading '{bucket}/{s3_file_path}'")
        raise  # re-raise last exception
    return object_content

from torch.nn.parallel import DistributedDataParallel as DDP
from models import VAR, VQVAE, build_vae_var
from var_trainer import VARTrainer
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params
    
vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=64, ch=128, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
)
# vae_ckpt 
vae_ckpt = get_file_s3()
vae_buffer = io.BytesIO(vae_ckpt)
checkpoint = torch.load(vae_buffer)
state_dict = checkpoint['state_dict']  # or however the checkpoint is structured

# Remove 'model.' prefix
new_state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}

# Load the adjusted state_dict into the model   
vae_local.load_state_dict(new_state_dict)
print("loaded")



