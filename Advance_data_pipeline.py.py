from albumentations import *
import cv2

class ImageAugment:
    def __init__(self, img_size=380):
        self.transform = Compose([
            RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(10, 15, 10),
            RandomBrightnessContrast(0.1, 0.1),
            Cutout(max_h_size=30, max_w_size=30, p=0.5),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            GaussNoise(var_limit=(10, 50)),
            ToGray(p=0.2),
            ImageCompression(quality_lower=60, p=0.3),
            Normalize()
        ])

class VideoAugment:
    def __init__(self, img_size=380):
        self.spatial_aug = ImageAugment(img_size)
        self.temporal_aug = TemporalTransform()  # Frame sampling

class TemporalTransform:
    """Video-specific augmentations"""
    def __call__(self, frames):
        # Frame sampling strategies
        if random.random() > 0.5:
            # Temporal dropout
            idx = sorted(random.sample(range(len(frames)), k=int(len(frames)*0.75)))
            frames = [frames[i] for i in idx]
        
        # Reverse temporal order
        if random.random() > 0.5:
            frames = frames[::-1]
            
        return frames