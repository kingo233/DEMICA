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
import face_segmentation_pytorch as fsp
from face_segmentation_pytorch.utils import normalize_range
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale


model = fsp.model.FaceSegmentationNet()
fsp.utils.load_model_parameters(model,'data')
model.eval()
model.cuda()

MEAN_BGR = np.array([104.00699, 116.66877, 122.67892])

def segment(img_array):
    # img: h,w,3
    img_array = cv2.resize(img_array,(500,500))
    img_array = (img_array * 255.0).astype(np.float32)
    # cv2.imwrite('test.jpg',img_array)
    img_array = normalize_range(img_array,out_range=(0, 255))
    img_array -= MEAN_BGR
    img_array = img_array.transpose((2, 0, 1))
    img = torch.from_numpy(img_array).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(img, as_pmap=True)[0]
        out = out.cpu().numpy()
    out = cv2.resize(out,(224,224))
    out[out >= 0.3] = 1
    out[out < 0.3] = 0
    # cv2.imwrite('test_out.jpg',out)
    return out


class BaseDataset(Dataset, ABC):
    def __init__(self, name, config, isEval):
        self.K = config.K
        self.isEval = isEval
        self.actors = []
        self.face_dict = {}
        self.name = name
        self.min_max_K = 0
        self.dataset_root = '/home/data3/czy3d/datasets/DEMICA_dataset'
        self.total_images = 0
        self.image_folder = 'images'
        self.flame_folder = 'FLAME_parameters'
        self.mask_folder = 'masks'
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list_file = f'{self.dataset_root}/image_paths/{str.upper(self.name)}.npy'
        if os.path.exists(image_list_file):
            logger.info(f'[{self.name}] Load cached file list: ' + image_list_file)
            self.face_dict = np.load(image_list_file, allow_pickle=True).item()
            self.actors = list(self.face_dict.keys())
        else:
            self.actors = os.listdir(os.path.join(self.dataset_root,self.name,self.image_folder))
            self.face_dict = {}
            for actor in self.actors:
                prefix_path = os.path.join(self.dataset_root,self.name,self.image_folder,actor)
                actor_img_list = os.listdir(prefix_path)
                actor_img_list = [os.path.join(actor,img_path) for img_path in actor_img_list]
                self.face_dict[actor] = actor_img_list
        logger.info(f'[Dataset {self.name}] Total {len(self.actors)} actors loaded!')
        self.set_smallest_k()
        # 为把原始图片裁剪成224*224做准备
        self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224))
        self.fan = FAN()
        self.scale = 1.25
    
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

    def set_smallest_k(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][0])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        return self.min_max_K

    def __len__(self):
        return len(self.actors)

    def __getitem__(self, index):
        actor = self.actors[index]
        if isinstance(self.face_dict[actor][0],list):
            images_path, params_path = self.face_dict[actor]
        else:
            images_path = self.face_dict[actor]
            params_path = 'none'
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

        flame_path = os.path.join(self.dataset_root, self.name, self.flame_folder, params_path)
        if os.path.exists(flame_path):
            params = np.load(flame_path, allow_pickle=True)
            pose = torch.tensor(params['pose']).float()
            betas = torch.tensor(params['betas']).float()

            flame = {
                'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
                'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
                'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
            }
        else:
            # logger.warning(f'{self.name} dataset have no flame parameters!Use None')
            flame = {}
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

            # MICA 部分
            # 以下部分是在跑一个人脸检测模型，得到置信分数最高的边界框，然后裁剪

            # arcface_path
            arcface_path = str(image_path).replace('images','arcface')
            if os.path.exists(arcface_path):
                arcface = cv2.imread(str(arcface_path))
                arcface = cv2.cvtColor(arcface,cv2.COLOR_BGR2RGB)
                arcface_list.append(arcface)
            else:
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
                arcface = cv2.resize(arcface,(112,112))
                arcface = np.transpose(arcface,(2,0,1))
                # 224 * 224对应images，是detail encoder输入
                arcface_list.append(arcface)

                cv2.imwrite(str(arcface_path),arcface)

            

            # DECA部分
            dst_path = str(image_path).replace('images','arcface')
            if os.path.exists(dst_path):
                dst_image = cv2.imread(dst_path)
                dst_image = cv2.cvtColor(dst_image,cv2.COLOR_BGR2RGB)
                image_list.append(dst_image)
            else:
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
                image_list.append(dst_image)

                cv2.imwrite(str(dst_path),dst_image)

            # 获得 landmarks，也即224*224的detail

            landmark_input = np.transpose(dst_image,(1,2,0))
            landmark = self.fan.model.get_landmarks(landmark_input * 255)
            # 归一到[-1,1]
            landmark[0] = landmark[0] / 224.0 * 2 - 1
            # 堆叠[68,2] ->[68,3]
            landmark[0] = np.hstack([landmark[0],np.ones([68,1],dtype=np.float32)])

            landmark_list.append(landmark[0])

            # 获得mask
            # dst_image = dst_image.transpose(1,2,0)
            # single_mask = segment()
            # single_mask = single_mask.reshape(224,224,1)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
                single_mask = np.zeros((mask.shape[0],mask.shape[1],1),dtype=np.uint8)
                cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY,single_mask)
            else:
                single_mask = np.zeros((224,224,1),dtype=np.int8)
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