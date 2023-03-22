import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from .detectors import FAN

class NoWDatasetBackup(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        folder = '/home/data3/czy3d/datasts/DEMICA_dataset/NoW_Dataset'
        self.data_path = os.path.join(folder, 'imagepathsvalidation.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'final_release_version', 'iphone_pictures')
        self.bbxfolder = os.path.join(folder, 'final_release_version', 'detected_face')

        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        # self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        # self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.crop_size = crop_size
        self.scale = scale
            
    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip()) #+ '.jpg'
        bbx_path = os.path.join(self.bbxfolder, self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']; right = bbx_data['right']
        top = bbx_data['top']; bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:,:,:3]

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = int(old_size*self.scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        arcface = cv2.resize(dst_image,(112,112))
        arcface = arcface.transpose(2,0,1)
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                'arcface': torch.tensor(arcface).float()
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }

class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        folder = '/home/data3/czy3d/datasets/DEMICA_dataset/NoW_Dataset'
        self.data_path = os.path.join(folder, 'imagepathsvalidation.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'final_release_version', 'iphone_pictures')
        # 为把原始图片裁剪成224*224做准备
        self.app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224))
        self.fan = FAN()

    def __len__(self):
        return len(self.data_lines)
    
    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip()) #+ '.jpg'

        img = imread(imagepath)[:,:,:3]
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
        detail_input = arcface
        arcface = cv2.resize(arcface,(112,112))
        arcface = np.transpose(arcface,(2,0,1))
        # 224 * 224对应images，是detail encoder输入
        detail_input = np.transpose(detail_input,(2,0,1))

        return {'image': torch.tensor(detail_input).float(),
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                'arcface': torch.tensor(arcface).float()
                }

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

import math
def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))