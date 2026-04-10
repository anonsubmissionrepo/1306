import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import re

def natural_sort_key(s):
    """Sorts strings containing numbers in a way that humans expect."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class ToTensor(object):
    """Converts PIL Images to Tensors and handles initial scaling."""
    def __call__(self, data):
        data['image'] = F.to_tensor(data['image'])
        data['label'] = F.to_tensor(data['label'])
        return data

class Resize(object):
    """Resizes image, label to a square resolution."""
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, data):
        data['image'] = F.resize(data['image'], self.size, interpolation=Image.BILINEAR)
        data['label'] = F.resize(data['label'], self.size, interpolation=Image.NEAREST)
        return data

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data['image'] = F.hflip(data['image'])
            data['label'] = F.hflip(data['label'])
        return data

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data['image'] = F.vflip(data['image'])
            data['label'] = F.vflip(data['label'])
        return data

class SAM3Normalize(object):
    """
    Normalizes RGB to SAM3 standards.
    SAM3 uses mean=0.5 and std=0.5 for images.
    """
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # Normalize RGB Image
        data['image'] = F.normalize(data['image'], self.mean, self.std)
        return data

class FullDataset(Dataset):
    def __init__(self, args, size, mode):
        if mode == 'train':
            image_root = os.path.join(args.path, "RGB/")
            gt_root = os.path.join(args.path, "GT/")
        else:
            image_root = os.path.join(args.val_path, "RGB/")
            gt_root = os.path.join(args.val_path, "GT/")
            
        self.size = size
        self.images = sorted([image_root + f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))], key=natural_sort_key)
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))], key=natural_sort_key)

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize(self.size),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                SAM3Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize(self.size),
                ToTensor(),
                SAM3Normalize()
            ])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.gts[idx]).convert('L')

        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

class TestDataset:
    def __init__(self, args, size):
        image_root = os.path.join(args.path, "RGB/")
        gt_root = os.path.join(args.path, "GT/")

        self.images = sorted([image_root + f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))], key=natural_sort_key)
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))], key=natural_sort_key)
        
        self.size = size
        self.length = len(self.images)
        self.index = 0

        # Normalization matching SAM3 processor
        self.img_transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def load_data(self):
        image = Image.open(self.images[self.index]).convert('RGB')
        gt = np.array(Image.open(self.gts[self.index]).convert('L'))

        # Prepare Image
        image = self.img_transform(image).unsqueeze(0)

        name = os.path.basename(self.images[self.index])
        self.index += 1

        return image, gt, name