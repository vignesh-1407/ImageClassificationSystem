"""
src/evaluate.py - Model evaluation: accuracy, F1, confusion matrix, loss curves
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocess import load_cifar10
from src.model import load_model


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="#30363d",
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", color="white", pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12, color="white")
    ax.set_ylabel("True Label", fontsize=12, color="white")
    ax.tick_params(colors="white", labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    """Bar chart of per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = ["#63b3ed" if a >= 0.75 else "#f6ad55" if a >= 0.5 else "#fc8181"
              for a in per_class_acc]

    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor="#30363d", linewidth=0.5)
    ax.axhline(y=np.mean(per_class_acc), color="#68d391", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(per_class_acc):.2%}")

    # Value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_xlabel("Class", fontsize=11, color="white")
    ax.set_ylabel("Accuracy", fontsize=11, color="white")
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#21262d", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Per-class accuracy chart saved → {save_path}")


def run_evaluation(model_path=None):
    """Load saved model and evaluate on CIFAR-10 test set."""
    print("\n" + "="*60)
    print("  AI IMAGE CLASSIFICATION SYSTEM — EVALUATION")
    print("="*60 + "\n")

    model_path = model_path or config.MODEL_PATH
    model = load_model(model_path)

    _, _, (x_test, y_test) = load_cifar10()

    print("[INFO] Running predictions on test set...")
    y_pred_probs = model.predict(x_test, batch_size=config.BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "─"*50)
    print(f"  Overall Accuracy  : {acc*100:.2f}%")
    print(f"  Weighted F1 Score : {f1*100:.2f}%")
    print(f"  Precision         : {precision*100:.2f}%")
    print(f"  Recall            : {recall*100:.2f}%")
    print("─"*50 + "\n")

    # Detailed per-class report
    print("[INFO] Detailed Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print(report)

    # Save report to file
    report_path = os.path.join(config.OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy  : {acc*100:.2f}%\n")
        f.write(f"Weighted F1 Score : {f1*100:.2f}%\n")
        f.write(f"Precision         : {precision*100:.2f}%\n")
        f.write(f"Recall            : {recall*100:.2f}%\n\n")
        f.write(report)
    print(f"[INFO] Report saved → {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm, config.CLASS_NAMES,
        os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    )
    plot_class_accuracy(
        y_true, y_pred, config.CLASS_NAMES,
        os.path.join(config.OUTPUTS_DIR, "per_class_accuracy.png")
    )

    print("\n[INFO] Evaluation complete. All outputs saved to:", config.OUTPUTS_DIR)
    return acc, f1


if __name__ == "__main__":
    run_evaluation()
