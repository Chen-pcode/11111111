from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        # 根据 train 标志选择 train 或 val 子目录。
        # images 与 masks 会排序后一一配对，因此文件命名必须能保证排序后顺序一致。
        if train:
            images_list = os.listdir(path_Data+'train/images/')
            masks_list = os.listdir(path_Data+'train/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(path_Data+'val/images/')
            masks_list = os.listdir(path_Data+'val/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        # 原图转 RGB，mask 转单通道灰度。
        # mask 除以 255 后变为 0/1 或 0-1 浮点标签，适配 BCE/Dice 损失。
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        # transformer 会同时处理 image 和 mask，保证翻转/旋转等空间增强保持对齐。
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        # DataLoader 用该长度决定每个 epoch 迭代多少个样本。
        return len(self.data)
        
    
