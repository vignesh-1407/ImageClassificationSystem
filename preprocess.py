"""
src/preprocess.py - Image preprocessing, augmentation, and dataset loading pipeline
"""

import numpy as np
import cv2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Normalization ──────────────────────────────────────────────────────────────
def normalize(images):
    """Normalize pixel values from [0, 255] to [0, 1]."""
    return images.astype("float32") / 255.0


# ── Resize (for custom images) ─────────────────────────────────────────────────
def resize_image(image, size=config.IMG_SIZE):
    """Resize a single image to target size using OpenCV."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_single_image(image_path, size=config.IMG_SIZE):
    """
    Load, resize, and normalize a single image from disk.
    Returns shape: (1, H, W, C) ready for model.predict()
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img, size)
    img = normalize(np.expand_dims(img, axis=0))  # (1, H, W, C)
    return img


# ── CIFAR-10 Dataset Loading ───────────────────────────────────────────────────
def load_cifar10():
    """
    Load CIFAR-10 dataset (auto-downloaded on first run).
    Returns preprocessed train/validation/test splits.
    """
    print("[INFO] Loading CIFAR-10 dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Normalize
    x_train_full = normalize(x_train_full)
    x_test       = normalize(x_test)

    # One-hot encode labels
    y_train_full = to_categorical(y_train_full, config.NUM_CLASSES)
    y_test       = to_categorical(y_test, config.NUM_CLASSES)

    # Split off validation set
    val_count   = int(len(x_train_full) * config.VALIDATION_SPLIT)
    x_val       = x_train_full[:val_count]
    y_val       = y_train_full[:val_count]
    x_train     = x_train_full[val_count:]
    y_train     = y_train_full[val_count:]

    print(f"[INFO] Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ── Class Weights for Balancing ────────────────────────────────────────────────
def compute_weights(y_train_onehot):
    """
    Compute class weights to handle class imbalance.
    y_train_onehot: one-hot encoded labels, shape (N, num_classes)
    """
    y_integers = np.argmax(y_train_onehot, axis=1)
    classes    = np.unique(y_integers)
    weights    = compute_class_weight(class_weight="balanced", classes=classes, y=y_integers)
    return dict(zip(classes, weights))


# ── Data Augmentation Generator ────────────────────────────────────────────────
def get_train_generator(x_train, y_train):
    """
    Build an ImageDataGenerator with augmentation for training data.
    """
    datagen = ImageDataGenerator(
        rotation_range=config.AUGMENT_ROTATION,
        zoom_range=config.AUGMENT_ZOOM,
        width_shift_range=config.AUGMENT_WIDTH_SHIFT,
        height_shift_range=config.AUGMENT_HEIGHT_SHIFT,
        horizontal_flip=config.AUGMENT_HORIZONTAL_FLIP,
        fill_mode="nearest"
    )
    datagen.fit(x_train)
    return datagen.flow(x_train, y_train, batch_size=config.BATCH_SIZE)


def get_val_generator(x_val, y_val):
    """Validation generator — no augmentation, only normalization (already done)."""
    datagen = ImageDataGenerator()
    return datagen.flow(x_val, y_val, batch_size=config.BATCH_SIZE, shuffle=False)
