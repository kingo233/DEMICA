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
from .Stirling import StirlingDataset

def build_train(config, is_train=True):
    data_list = []
    if 'Stirling' in config.training_data:
        data_list.append(StirlingDataset(config))
    dataset = ConcatDataset(data_list)
    
    return dataset

def build_val(config, is_train=True):
    data_list = []
    if 'now' in config.eval_data:
        data_list.append(NoWDataset())
    dataset = ConcatDataset(data_list)

    return dataset
    