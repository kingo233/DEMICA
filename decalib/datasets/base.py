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
import cv2
import math
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from .detectors import FAN
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
        self.dataset_root = 'dataset'
        self.total_images = 0
        self.image_folder = 'images'
        self.flame_folder = 'FLAME_parameters'
        self.mask_folder = 'masks'
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        # 只要是数据集都会有npy文件
        image_list_file = f'dataset/image_paths/{str.upper(self.name)}.npy'
        logger.info(f'[{self.name}] Load cached file list: ' + image_list_file)
        self.face_dict = np.load(image_list_file, allow_pickle=True).item()
        self.actors = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.actors)} actors loaded!')
        self.set_smallest_k()
        # 为把原始图片裁剪成224*224做准备
        self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224))
        self.fan = FAN()

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
        images_path, params_path = self.face_dict[actor]
        # 根据数据集的不同决定要不要把actor前缀消除
        if self.name == 'Stirling':
            images_path = [path.split('/')[1] for path in images_path]
        images_path = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images_path]

        # 把images字样换成masks就是mask路径
        masks_path = [str(path).replace('images','masks') for path in images_path]

        sample_list = np.array(np.random.choice(range(len(images_path)), size=self.K, replace=False))

        K = self.K
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images_path))[:K])

        params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path), allow_pickle=True)
        pose = torch.tensor(params['pose']).float()
        betas = torch.tensor(params['betas']).float()

        flame = {
            'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
            'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
            'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
        }

        arcface_list = []
        landmark_list = []
        mask_list = []
        image_list = []

        for i in sample_list:
            image_path = images_path[i]
            mask_path = masks_path[i]
            # w * h * 3
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # 以下部分是在跑一个人脸检测模型，得到置信分数最高的边界框，然后裁剪
            bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                continue
            i = get_center(bboxes, img)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            # 获得裁剪的112*112输出，arcface用于粗糙模型，对应arcface
            arcface = face_align.norm_crop(img, landmark=face.kps, image_size=224)
            arcface = arcface / 255.0
            detail_input = arcface
            arcface = cv2.resize(arcface,(112,112))
            arcface = np.transpose(arcface,(2,0,1))
            # 224 * 224对应images，是detail encoder输入
            detail_input = np.transpose(detail_input,(2,0,1))
            image_list.append(detail_input)
            arcface_list.append(arcface)

            # 获得 landmarks，针对粗糙模型的，也即112*112的arcface
            arcface = np.transpose(arcface,(1,2,0))
            landmark = self.fan.model.get_landmarks(arcface * 255)
            # 归一到[-1,1]
            landmark[0] = landmark[0] / 112 * 2 - 1
            # 堆叠[68,2] ->[68,3]
            landmark[0] = np.hstack([landmark[0],np.ones([68,1],dtype=np.float32)])

            landmark_list.append(landmark[0])

            # 获得mask
            mask = cv2.imread(mask_path)
            single_mask = np.zeros((mask.shape[0],mask.shape[1],1),dtype=np.uint8)
            cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY,single_mask)
            single_mask = cv2.resize(single_mask,(224,224))
            single_mask[single_mask > 0] = 1
            mask_list.append(np.array(single_mask))

        arcfaces_array = torch.from_numpy(np.array(arcface_list)).float()
        landmarks = torch.from_numpy(np.array(landmark_list)).float()
        masks = torch.from_numpy(np.array(mask_list)).float()
        images = torch.from_numpy(np.array(image_list)).float()

        return {
            # images->224 * 224,是detail部分的输入
            'images': images,
            'arcface': arcfaces_array,
            'imagename': actor,
            'dataset': self.name,
            'flame': flame,
            'landmark':landmarks,
            'mask':masks
        }


def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_center(bboxes, img):
    img_center = img.shape[0] // 2, img.shape[1] // 2
    size = bboxes.shape[0]
    distance = np.Inf
    j = 0
    for i in range(size):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx = abs(x2 - x1) / 2.0
        dy = abs(y2 - y1) / 2.0
        current = dist((x1 + dx, y1 + dy), img_center)
        if current < distance:
            distance = current
            j = i

    return j

input_mean = 127.5
input_std = 127.5


def get_arcface_input(face, img):
    aimg = face_align.norm_crop(img, landmark=face.kps)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0], aimg