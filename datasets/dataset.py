from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from research_support import paired_files


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        split = 'train' if train else 'val'
        self.data = [[str(image), str(mask)] for _, image, mask in paired_files(path_Data, split)]
        self.transformer = config.train_transformer if train else config.test_transformer
        
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
        
    
