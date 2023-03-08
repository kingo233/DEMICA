import os, sys
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from .now import NoWDataset
# from .Stirling import StirlingDataset
from .base import BaseDataset

def build_train(config, is_train=True):
    data_list = []
    for name in config.training_data:
        data_list.append(BaseDataset(name,config,False))
    dataset = ConcatDataset(data_list)
    
    return dataset

def build_val(config, is_train=True):
    data_list = []
    config.K = 1
    for name in config.eval_data:
        data_list.append(BaseDataset(name,config,False))
    dataset = ConcatDataset(data_list)

    return dataset
    