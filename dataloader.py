import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random


# Create custom PyTorch for files 
class CellDataset(Dataset):
    # Images and masks are stored in imgs and masks
    def __init__(self, imgs, masks, args = None):
        self.imgs = imgs
        self.masks = masks
        self.args = args
        
        # Random batch transforms
        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomCrop(args.patch_size),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.Resize(args.patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.default_transform = transforms.Compose([
            transforms.Resize(args.patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.ToTensor()
        
    # Return number of samples in the dataset
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        
        # Apply augmentation with a certain probability
        if self.args.augfly and random.random() < 0.5:  # 50% probability to apply augmentation
            img = self.augmentation_transform(img)
        else:
            img = self.default_transform(img)
        
        mask = self.mask_transform(mask)
        
        return img, mask
