# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import re
from abc import ABC
from functools import reduce
from pathlib import Path

import loguru
import numpy as np
import torch
from loguru import logger
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset, ABC):
    def __init__(self, name, config, isEval):
        self.K = config.K
        self.isEval = isEval
        self.actors = []
        self.face_dict = {}
        self.name = name
        self.min_max_K = 0
        self.dataset_root = config.root
        self.total_images = 0
        self.image_folder = 'arcface_input'
        self.flame_folder = 'FLAME_parameters'
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list_file = f'{os.path.abspath(os.path.dirname(__file__))}/image_paths/{self.name}.npy'
        logger.info(f'[{self.name}] Load cached file list: ' + image_list_file)
        self.face_dict = np.load(image_list_file, allow_pickle=True).item()
        self.actors = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.actors)} actors loaded!')
        self.set_smallest_k()

    def set_smallest_k(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][0])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][0]), self.actors))
        loguru.logger.info(f'Dataset {self.name} with min K = {self.min_max_K} max K = {max_min_k} length = {len(self.face_dict)} total images = {self.total_images}')
        return self.min_max_K

    def __len__(self):
        return len(self.actors)

    def __getitem__(self, index):
        actor = self.actors[index]
        images, params_path = self.face_dict[actor]
        # 把actor前缀消除
        images = [path.split('/')[1] for path in images]
        images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]
        sample_list = np.array(np.random.choice(range(len(images)), size=self.K, replace=False))

        K = self.K
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path), allow_pickle=True)
        pose = torch.tensor(params['pose']).float()
        betas = torch.tensor(params['betas']).float()

        flame = {
            'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
            'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
            'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
        }

        images_list = []
        arcface_list = []

        for i in sample_list:
            image_path = images[i]
            image = np.array(imread(image_path))
            image = image / 255.
            image = image.transpose(2, 0, 1)

            images_list.append(image)

        images_array = torch.from_numpy(np.array(images_list)).float()

        return {
            'image': images_array,
            'imagename': actor,
            'dataset': self.name,
            'flame': flame,
            # 'arcface':arcfaces
            # 'landmark':lmks
            # 'mask':masks
        }
