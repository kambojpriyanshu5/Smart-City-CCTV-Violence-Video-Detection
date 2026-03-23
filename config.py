import os
from pathlib import Path

DATA_PATH = Path("D:\SCVD dataset")
TRAIN_PATH = DATA_PATH / "train"
VAL_PATH = DATA_PATH / "val"
TEST_PATH = DATA_PATH / "test"

# Model hyperparameters
IMG_SIZE = 224
NUM_FRAMES = 8
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 3

# Match your actual folder names
CLASSES = ['Normal', 'Violence', 'Weaponized']  # For confusion matrix labels
CLASS_TO_IDX = {'Normal': 0, 'Violence': 1, 'Weaponized': 2}

# Output folder
RESULTS_PATH = Path("results/")
MODEL_PATH = RESULTS_PATH / "scvd_violence_model.pth"
