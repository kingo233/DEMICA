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
import os

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
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale

class REALYDataset(Dataset):
    def __init__(self):
        self.name = 'REALY'
        self.dataset_root = '/home/data3/czy3d/datasets/REALY_benchmark/REALY_image/crop_image_frontal_512x512'
        self.total_images = 0
        self.initialize()

    def initialize(self):
        # 为把原始图片裁剪成224*224做准备
        self.app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224))
        self.fan = FAN()
        self.scale = 1.25
        self.actors = os.listdir(self.dataset_root)
    
    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center


    def __len__(self):
        return len(self.actors)

    def __getitem__(self, index):
        actor = self.actors[index]

        image_path = os.path.join(self.dataset_root,actor)
        # w * h * 3
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # MICA 部分
        # 以下部分是在跑一个人脸检测模型，得到置信分数最高的边界框，然后裁剪

 
        bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
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
        arcface = cv2.resize(arcface,(112,112))
        arcface = np.transpose(arcface,(2,0,1))
        

        
        image = np.array(imread(str(image_path)))
        h, w, _ = image.shape
        bbox, bbox_type = self.fan.run(image)
        if len(bbox) < 4:
            print('no face detected! run original image')
            left = 0; right = h-1; top=0; bottom=w-1
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size*self.scale)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        
        self.resolution_inp = 224
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)


        return {
            'image': torch.from_numpy(dst_image).float(),
            'arcface': torch.from_numpy(arcface).float(),
            'imagename': actor.replace('.jpg','')
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
