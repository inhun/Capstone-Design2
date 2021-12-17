from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image



def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=256):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_size = img_size
            
        self.label_files = [
            path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
            for path in self.img_files
        ]

        self.img_size = img_size

    
    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path, 'r').convert('RGB'))

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expend((3, img.shape[1:]))
        
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            targets = torch.from_numpy(np.loadtxt(label_path)).long()
            

        return img_path, img, targets


    def collant_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        targets = [target for target in targets]
        targets = torch.tensor(targets)
        return paths, imgs, targets


    def __len__(self):
        return len(self.img_files)