# config.py

import os
import torch
# Base data path
DATA_DIR = r"D:\deepfake_video_dataset"  

# Data subdirectories
TRAIN_DIR = os.path.join(DATA_DIR, r"train")
VAL_DIR = os.path.join(DATA_DIR, r"val")
TEST_DIR = os.path.join(DATA_DIR, r"test")  

# Checkpoint path
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model save path
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "deepfake_model.pth")

# Model config
NUM_CLASSES = 2
USE_TEMPORAL = True
IMG_SIZE = 256

# Training config
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
BACKBONE_LR = 1e-5
TEMPORAL_LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2 

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
PRINT_FREQ = 10
SAVE_FREQ = 1
