"""
src/model.py - CNN architecture definition
3-block Convolutional Neural Network with BatchNorm and Dropout
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, GlobalAveragePooling2D, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def build_cnn(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
              num_classes=config.NUM_CLASSES,
              learning_rate=config.LEARNING_RATE):
    """
    Build and compile the CNN model.

    Architecture:
        Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.2)
        Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.3)
        Block 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.4)
        Head:    Flatten → Dense(512) → BN → ReLU → Dropout(0.5) → Dense(num_classes, Softmax)
    """
    model = Sequential(name="CNN_ImageClassifier")

    # ── Block 1 ────────────────────────────────────────────────────────────────
    model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-4),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # ── Block 2 ────────────────────────────────────────────────────────────────
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # ── Block 3 ────────────────────────────────────────────────────────────────
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # ── Classifier Head ────────────────────────────────────────────────────────
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(config.DROPOUT_RATE_2))
    model.add(Dense(num_classes, activation="softmax"))

    # ── Compile ────────────────────────────────────────────────────────────────
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def load_model(path=config.MODEL_PATH):
    """Load a saved Keras model from disk."""
    from tensorflow.keras.models import load_model as keras_load
    print(f"[INFO] Loading model from: {path}")
    return keras_load(path)


def model_summary(model):
    """Print a summary of the model architecture."""
    model.summary()
    total = model.count_params()
    print(f"\n[INFO] Total parameters: {total:,}")
