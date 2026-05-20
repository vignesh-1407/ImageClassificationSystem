"""
config.py - Central configuration for the AI Image Classification System
All hyperparameters, paths, and settings are defined here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
SAMPLE_DIR      = os.path.join(BASE_DIR, "sample_images")
MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.h5")
FINAL_MODEL_PATH= os.path.join(MODELS_DIR, "final_model.h5")

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET         = "cifar10"           # "cifar10" | "custom"
CUSTOM_DATA_DIR = os.path.join(DATA_DIR, "custom")  # used when DATASET="custom"

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Image Settings ─────────────────────────────────────────────────────────────
IMG_HEIGHT  = 32
IMG_WIDTH   = 32
IMG_CHANNELS= 3
IMG_SIZE    = (IMG_HEIGHT, IMG_WIDTH)

# ── Training Hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE      = 64
EPOCHS          = 30
LEARNING_RATE   = 0.001
VALIDATION_SPLIT= 0.1

# ── Augmentation Settings ─────────────────────────────────────────────────────
AUGMENT_ROTATION    = 15        # degrees
AUGMENT_ZOOM        = 0.1
AUGMENT_WIDTH_SHIFT = 0.1
AUGMENT_HEIGHT_SHIFT= 0.1
AUGMENT_HORIZONTAL_FLIP = True

# ── Early Stopping ────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 8
LR_REDUCE_PATIENCE  = 4
LR_REDUCE_FACTOR    = 0.5
MIN_LR              = 1e-6

# ── Model Architecture ────────────────────────────────────────────────────────
DROPOUT_RATE_1  = 0.3
DROPOUT_RATE_2  = 0.5

# ── Prediction ────────────────────────────────────────────────────────────────
TOP_K_PREDICTIONS   = 3          # show top-K classes in prediction output
CONFIDENCE_THRESHOLD= 0.50       # minimum confidence to display prediction

# Create directories
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SAMPLE_DIR]:
    os.makedirs(d, exist_ok=True)
