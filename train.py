"""
src/train.py - Full training pipeline with callbacks and model checkpointing
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocess import load_cifar10, compute_weights, get_train_generator, get_val_generator
from src.model import build_cnn, model_summary

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)


def get_callbacks():
    """Define training callbacks for checkpointing, early stopping, LR scheduling."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    callbacks = [
        # Save best model by val_accuracy
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Stop early if no improvement
        EarlyStopping(
            monitor="val_accuracy",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce LR on plateau
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        # CSV log
        CSVLogger(os.path.join(config.OUTPUTS_DIR, "training_log.csv"))
    ]
    return callbacks


def plot_training_history(history):
    """Save accuracy and loss curves to outputs/."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     color="#63b3ed", linewidth=2, label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], color="#68d391", linewidth=2, label="Val Accuracy",  linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#21262d", labelcolor="white")
    axes[0].grid(color="#30363d", linewidth=0.5)

    # Loss
    axes[1].plot(history.history["loss"],     color="#f6ad55", linewidth=2, label="Train Loss")
    axes[1].plot(history.history["val_loss"], color="#fc8181", linewidth=2, label="Val Loss",  linestyle="--")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#21262d", labelcolor="white")
    axes[1].grid(color="#30363d", linewidth=0.5)

    plt.tight_layout()
    out_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Training curves saved → {out_path}")


def run_training():
    """Main training function."""
    print("\n" + "="*60)
    print("  AI IMAGE CLASSIFICATION SYSTEM — TRAINING")
    print("="*60 + "\n")

    # 1. Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10()

    # 2. Class weights for balanced training
    class_weights = compute_weights(y_train)
    print(f"[INFO] Class weights: { {config.CLASS_NAMES[k]: round(v, 3) for k, v in class_weights.items()} }")

    # 3. Data generators
    train_gen = get_train_generator(x_train, y_train)
    val_gen   = get_val_generator(x_val, y_val)

    # 4. Build model
    model = build_cnn()
    model_summary(model)

    # 5. Train
    steps_per_epoch  = len(x_train) // config.BATCH_SIZE
    validation_steps = len(x_val)   // config.BATCH_SIZE

    print(f"\n[INFO] Training for up to {config.EPOCHS} epochs...")
    start = time.time()

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=get_callbacks(),
        class_weight=class_weights,
        verbose=1
    )

    elapsed = time.time() - start
    print(f"\n[INFO] Training completed in {elapsed/60:.1f} minutes.")

    # 6. Save final model
    model.save(config.FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved → {config.FINAL_MODEL_PATH}")

    # 7. Plot
    plot_training_history(history)

    # 8. Quick test eval
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] Test Accuracy : {test_acc*100:.2f}%")
    print(f"[RESULT] Test Loss     : {test_loss:.4f}")

    return history, model


if __name__ == "__main__":
    run_training()
