# Advanced AI-Based Image Classification System

A production-quality CNN image classification system built with **Python**, **TensorFlow**, **Keras**, **OpenCV**, and **NumPy**.

---

## Features

- ✅ **3-Block CNN** with BatchNorm, Dropout, L2 regularization
- ✅ **Image Preprocessing** — resize, normalize, augment, balance dataset
- ✅ **Training Pipeline** — EarlyStopping, LR scheduling, model checkpointing
- ✅ **Evaluation** — Accuracy, F1, Precision, Recall, Confusion Matrix, per-class charts
- ✅ **Real-Time Prediction** — single image or live webcam inference
- ✅ **CLI** — single entry point for all modes
- ✅ **Built with Antigravity Platform**

---

## Project Structure

```
ImageClassificationSystem/
├── data/                   # Dataset (auto-created)
├── models/                 # Saved models (.h5)
├── outputs/                # Plots, reports
├── sample_images/          # Test images
├── src/
│   ├── preprocess.py       # Data loading & augmentation
│   ├── model.py            # CNN architecture
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation & metrics
│   └── predict.py          # Inference (image / webcam)
├── main.py                 # CLI entry point
├── config.py               # Hyperparameters & paths
├── requirements.txt
├── setup.bat               # Windows setup
└── run.bat                 # Windows run menu
```

---

## Quick Start (Windows)

### 1. Setup
```bat
setup.bat
```
This creates a virtual environment and installs all dependencies.

### 2. Run
```bat
run.bat
```
An interactive menu lets you choose any mode.

---

## CLI Usage

```bash
# Train the CNN on CIFAR-10
python main.py --mode train

# Evaluate saved model
python main.py --mode evaluate

# Predict a single image
python main.py --mode predict --image sample_images/cat.jpg

# Real-time webcam prediction
python main.py --mode predict --webcam

# Print model architecture summary
python main.py --mode summary

# Override epochs
python main.py --mode train --epochs 10
```

---

## CNN Architecture

```
Input (32×32×3)
  │
  ├── Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.2)
  ├── Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.3)
  ├── Block 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.4)
  │
  └── Head: Flatten → Dense(512) → BN → ReLU → Dropout(0.5) → Dense(10, Softmax)
```

**Optimizer**: Adam | **Loss**: Categorical Cross-Entropy | **Regularization**: L2(1e-4)

---

## Dataset — CIFAR-10

| Class | Label |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

50,000 training images · 10,000 test images · 32×32 pixels

---

## Outputs

After training and evaluation, the `outputs/` folder contains:

| File | Description |
|---|---|
| `training_curves.png` | Accuracy & loss curves |
| `confusion_matrix.png` | Full confusion matrix heatmap |
| `per_class_accuracy.png` | Per-class accuracy bar chart |
| `classification_report.txt` | Precision, Recall, F1 per class |
| `training_log.csv` | Epoch-by-epoch metrics |

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `EPOCHS` | 30 | Max training epochs |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `IMG_SIZE` | (32, 32) | Input image size |
| `EARLY_STOP_PATIENCE` | 8 | Epochs before early stop |
| `TOP_K_PREDICTIONS` | 3 | Classes shown in prediction |

---

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- OpenCV 4.7+
- NumPy, scikit-learn, matplotlib, seaborn, Pillow

---

*Built with the **Antigravity Platform** for AI-assisted development.*
