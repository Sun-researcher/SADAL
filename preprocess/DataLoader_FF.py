# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
import random
import cv2
import torchvision.transforms.functional as F
from PIL import Image
import sys
import albumentations as alb
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from glob import glob
import json
import numpy as np
from PIL import Image
import os

def init_ff_fake(phase, n_frames=8, included_path=['Deepfakes', 'Face2Face', 'FaceSwap','NeuralTextures']):
    dataset_path_fake = 'data/FaceForensics++/manipulated_sequences/*/c23/frames/'  # 改动过，raw改为c23
    image_list = []
    label_list = []
    folder_list_fake = sorted(glob(dataset_path_fake + '*'))
    list_dict = json.load(open(f'data/FaceForensics++/{phase}.json', 'r'))

    filelist = []
    for i in list_dict:
        filelist += i
    print('Proportion of training datasets (in parts per thousand)', len(filelist))

    folder_list_fake = [i for i in folder_list_fake if os.path.basename(i)[:3] in filelist]
    for path in included_path:
        for folder in folder_list_fake:
            if path in folder:
                images_temp_fake = sorted(glob(folder + '/*.png'))

                if n_frames < len(images_temp_fake):  # 选取帧数
                    images_temp_fake = [images_temp_fake[round(i)] for i in
                                        np.linspace(0, len(images_temp_fake) - 1, n_frames)]
                image_list += images_temp_fake
                label_list += [1] * len(images_temp_fake)

    print('total pictures：', len(image_list))

    for i in range(len(image_list)):
        image_list[i] = image_list[i].replace('\\', '/')

    random.shuffle(image_list)
    # print(image_list)
    return image_list, label_list
def init_ff_real(phase, n_frames=8):
    dataset_path = 'data/FaceForensics++/original_sequences/youtube/c23/frames/'  # 改动过，raw改为c23

    image_list = []
    label_list = []
    folder_list = sorted(glob(dataset_path + '*'))
    list_dict = json.load(open(f'data/FaceForensics++/{phase}.json', 'r'))

    filelist = []
    for i in list_dict:
        filelist += i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    for i in range(len(folder_list)):
        images_temp = sorted(glob(folder_list[i] + '/*.png'))

        if n_frames < len(images_temp):
            images_temp = [images_temp[round(i)] for i in np.linspace(0, len(images_temp) - 1, n_frames)]
        image_list += images_temp
        label_list += [0] * len(images_temp)

    for i in range(len(image_list)):
        image_list[i] = image_list[i].replace('\\', '/')
    random.shuffle(image_list)
    # print(image_list)
    return image_list, label_list

class DataLoader_multi_source_and_target(Dataset):
    def __init__(self, phase='train',domain='source', image_size=224, n_frames=8, forgery=['Deepfakes', 'Face2Face', 'FaceSwap','NeuralTextures']):
        assert phase in ['train', 'val', 'test',]
        self.domain = domain
        real_image_list, real_label_list = init_ff_real(phase, n_frames=n_frames)
        fake_image_list, fake_label_list = init_ff_fake(phase, n_frames=n_frames, included_path=forgery)

        print(f'Real samples ({phase}): {len(real_image_list)}')
        print(f'Fake samples ({phase}): {len(fake_image_list)}')

        self.real_image_list = real_image_list
        self.real_label_list = real_label_list
        self.fake_image_list = fake_image_list
        self.fake_label_list = fake_label_list

        self.image_size = (image_size, image_size)
        self.phase = phase
        self.n_frames = n_frames
        self.num_forgery_types = len(fake_image_list)//len(real_image_list)

        self.source_transforms = self.get_source_transforms()
        self.target_transforms = self.get_target_transforms()
        self.get_transforms = self.get_transforms(image_size)
        self.get_augs = self.get_augs(image_size)

    def __len__(self):
        return min(len(self.real_image_list), len(self.fake_image_list) // self.num_forgery_types)

    def __getitem__(self, idx):
        real_idx = idx % len(self.real_image_list)
        real_image_path = self.real_image_list[real_idx]
        real_image = np.array(Image.open(real_image_path))
        real_label = self.real_label_list[real_idx]

        fake_start_idx = idx * self.num_forgery_types
        fake_images = []
        fake_labels = []

        for i in range(self.num_forgery_types):
            fake_idx = (fake_start_idx + i) % len(self.fake_image_list)
            fake_image_path = self.fake_image_list[fake_idx]
            fake_image = np.array(Image.open(fake_image_path))
            fake_images.append(fake_image)
            fake_labels.append(self.fake_label_list[fake_idx])

        if self.phase == 'train' and self.domain=='source':
            real_image = self.source_transforms(image=real_image.astype('uint8'))['image']
            fake_images = [self.source_transforms(image=img.astype('uint8'))['image'] for img in fake_images]
        if self.phase == 'train' and self.domain=='target':
            real_image = self.target_transforms(image=real_image.astype('uint8'))['image']
            fake_images = [self.target_transforms(image=img.astype('uint8'))['image'] for img in fake_images]

        if self.phase == 'train' and self.domain=='target':
            if np.random.rand() < 0.5:
                real_image = real_image[:, ::-1]
                fake_images = [img[:, ::-1] for img in fake_images]
            else:
                real_image = np.flipud(real_image)
                fake_images = [np.flipud(img) for img in fake_images]

        if self.phase == 'train' and self.domain=='target':
            real_image = self.get_augs(Image.fromarray(real_image).convert('RGB'))
            fake_images = [self.get_augs(Image.fromarray(img).convert('RGB')) for img in fake_images]
        else:
            real_image = self.get_transforms(Image.fromarray(real_image).convert('RGB'))
            fake_images = [self.get_transforms(Image.fromarray(img).convert('RGB')) for img in fake_images]

        images = [real_image] + fake_images
        labels = [real_label] + fake_labels

        return torch.stack(images), torch.tensor(labels)

    def get_transforms(self, size):
        IMG_SIZE = size
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomApply([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 0.8), ratio=(0.9, 1.1)),  # 加强
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_augs(self, size):
        IMG_SIZE = size
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomApply([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 0.5), ratio=(0.9, 1.1)),
            ], p=0.8),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.10, 0.20), ratio=(0.5, 2.0), inplace=True),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
                alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
            ], p=1),

        ], p=1.)

    def get_target_transforms(self):
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
        ],
            p=1.)

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def collate_fn(self,batch):
        images, labels = zip(*batch)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        data = {
            'img': images.float(),
            'label': labels.float()
        }
        return data


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 使用多种伪造类别
    image_dataset = DataLoader_multi_source_and_target(phase='train', image_size=256, n_frames=8,
                                                         forgery=['Deepfakes', 'Face2Face', 'FaceSwap','NeuralTextures'])
    batch_size = 16
    dataloader = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=image_dataset.collate_fn,
                                             num_workers=4,
                                             worker_init_fn=image_dataset.worker_init_fn)

    print(f'Number of batches: {len(dataloader)}')
    data_iter = iter(dataloader)
    data = next(data_iter)
    img = data['img']
    print(f'Batch image shape: {img.shape}')
    print(f'Labels: {data["label"]}')

