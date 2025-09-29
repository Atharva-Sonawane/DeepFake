#data.py

import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from albumentations import (
    Resize, Normalize, HorizontalFlip, RandomBrightnessContrast,
    CoarseDropout, ShiftScaleRotate, GaussianBlur, HueSaturationValue, Compose
)
from albumentations.pytorch import ToTensorV2

# Optional: For reproducibility
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")  # Ensures all images are RGB
        image = np.array(image)

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented["image"]

        return image, target

def get_transforms(phase, img_size):
    if phase == "train":
        return Compose([
            Resize(img_size, img_size),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.4),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            GaussianBlur(blur_limit=(3, 5), p=0.2),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=0.4),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(img_size, img_size),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

def create_dataloaders(train_dir, val_dir, batch_size, num_workers, img_size):
    train_dataset = AlbumentationsDataset(
        root=train_dir,
        transform=get_transforms("train", img_size)
    )
    val_dataset = AlbumentationsDataset(
        root=val_dir,
        transform=get_transforms("val", img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
