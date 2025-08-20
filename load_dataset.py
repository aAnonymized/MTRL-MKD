import torch
import os
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
from skimage import transform
import numpy as np
from torch import nn
import SimpleITK as sitk
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import nibabel as nib

import albumentations as A
from albumentations.pytorch import ToTensorV2

class ACDC(torch.utils.data.Dataset):
    def __init__(self, data=None, phase = 'train', img_size=(224, 224)):
        self.img_size = img_size
        self.datas = data
     
        ### 输入是要(w, h, c)
        self.seq = iaa.Sequential([
            iaa.PadToFixedSize(width=210, height=210, position="center"),
            iaa.Resize({"height": self.img_size[0], "width": self.img_size[0]}), 
            iaa.Affine(rotate=(-45, 45), order=1), 
            iaa.GaussianBlur(sigma=(0.1, 0.25)),
            iaa.LinearContrast((0.5, 1.5)), 
            iaa.Multiply((0.5, 1.5)),
        ], random_order=False)

    def __len__(self):
        return len(self.datas)
        
    def pad_or_truncate_T(self, video_array, target_frames=25):
        C, T, H, W = video_array.shape

        if T == target_frames:
            return video_array

        elif T > target_frames:
            return video_array[:, :target_frames, :, :]

        else:
            pad_frames = target_frames - T
            repeat_times = (pad_frames + T - 1) // T + 1
            repeated_frames = np.tile(video_array[:, :T, :, :], (1, repeat_times, 1, 1))[:, :pad_frames + T, :, :]
            return repeated_frames  # 3 25 224 224

    def __getitem__(self, index):
        fpath = self.datas.iloc[index]['full_path']
        linux_path = fpath.replace("\\", "/")
        image = nib.load(linux_path)
        image_array = image.get_fdata()  ## W C T _ H  (104, 3, 15, 1, 101)
        image_array=np.transpose(image_array, (3, 2, 0, -1, 1))[0]  ## T W H C
        seq_det = self.seq.to_deterministic()
        image_list = seq_det(images=image_array.astype(np.float32))
        image_array = np.array(image_list) # 拼接成数组
        image_array=np.transpose(image_array, (3, 0, 2, 1))  ## C T H W
        if image_array.shape[0] != 3 or image_array.shape[1] < 5 or image_array.shape[2] != 224 or image_array.shape[3] != 224:
            print(f"problem data : {fpath}")
            print(f'image_array {image_array.shape}')
        image_array = self.pad_or_truncate_T(image_array)
        image_array = image_array[:, ::2, :, :]
        image_tensor = torch.from_numpy(image_array).float()
        label = self.datas.iloc[index]['target_vector']
        label = label.astype(np.int64)
        label = torch.from_numpy(label).float()
        return image_tensor, self.datas.iloc[index]['Finding Labels'], label
