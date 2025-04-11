# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE
import json
from glob import glob
import os
import numpy as np
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')
from funcs import IoUfrom2bboxes, crop_face, RandomDownScale
from tqdm import tqdm
def save_FF(forgery='FaceShifter',img_size=(224,224),n_frames=2):
    # dataset_path_fake = 'data/FaceForensics++/original_sequences/*/c23/frames/'
    dataset_path_fake = 'data/FaceForensics++/manipulated_sequences/*/c23/frames/'
    image_list = []
    label_list = []
    #修改假图片路径
    dataset_path_fake = dataset_path_fake.replace('*', forgery)
    folder_list_fake = sorted(glob(dataset_path_fake + '*'))
    print(len(folder_list_fake))
    for i in range(len(folder_list_fake)):
        # image_list += images_temp_fake
        image_list += sorted(glob(folder_list_fake[i] + '/*.png'))
        label_list += [0] * len(image_list)


    print('总图像数量：',len(image_list))
    for i in range(len(image_list)):
        image_list[i] = image_list[i].replace('\\', '/')
    path_lm = '/landmarks/'
    path_retina = '/retina/'
    value_list = []
    valid_images = []
    for image in image_list:

        lm_file = image.replace('/frames/', path_lm).replace('.png', '.npy')
        retina_file = image.replace('/frames/', path_retina).replace('.png', '.npy')

        if os.path.isfile(lm_file) and os.path.isfile(retina_file):
            valid_images.append(image)
            value_list += [1]
    image_list = valid_images
    for idx in tqdm(range(len(image_list))):
        filename = image_list[idx]
        label=label_list[idx]
        maskfile = filename.replace('/frames/', '/masks/')
        img = np.array(Image.open(filename))
        if label == 1:
            img_mask = np.array(Image.open(maskfile))
        else:
            img_mask = np.zeros(img.shape)
        landmark = np.load(filename.replace('.png', '.npy').replace('/frames/', '/landmarks/'))[0]
        bbox_lm = np.array(
            [landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(),
             landmark[:, 1].max()])
        bboxes = np.load(filename.replace('.png', '.npy').replace('/frames/', '/retina/'))[:2]
        iou_max = -1
        for i in range(len(bboxes)):
            iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
            if iou_max < iou:
                bbox = bboxes[i]
                iou_max = iou

        # landmark = reorder_landmark(landmark)
        img, img_mask, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(img, img_mask, landmark, bbox,
                                                                              margin=False,
                                                                              crop_by_bbox=True, abs_coord=True,
                                                                              )
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
        save_path=filename.replace('/FaceForensics++/','/FF++/')
        target_directory = os.path.dirname(save_path)
        os.makedirs(target_directory, exist_ok=True)
        # Convert the numpy array to a PIL Image
        img_pil = Image.fromarray(img)

        # Save the PIL Image
        img_pil.save(save_path)

if __name__ == '__main__':
    save_FF(forgery='Face2Face',img_size=(224,224))


