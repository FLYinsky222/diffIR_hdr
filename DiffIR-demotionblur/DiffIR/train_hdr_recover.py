# flake8: noqa
import os.path as osp
import os
os.environ["TORCH_HOME"] = "/home/ubuntu/data_sota_disk/TORCH_HOME"
import torch
from DiffIR.train_pipeline_hdr_recover import train_pipeline_hdr_recover

import DiffIR.archs
import DiffIR.data
import DiffIR.models
import DiffIR.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline_hdr_recover(root_path)