# 🧠 CNN Image Classifier — TensorFlow/Keras

A clean, intermediate-level **Convolutional Neural Network** for image classification, trained on **CIFAR-10** with a fully custom training loop, cosine decay with warmup, data augmentation, and rich evaluation outputs.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Custom Training Loop** | `tf.GradientTape` — full control over gradient updates |
| **LR Schedule** | Cosine Decay with Linear Warmup |
| **Optimizer** | AdamW with weight decay |
| **Metrics** | Top-1 & Top-5 accuracy (train + val), per-class report |
| **Data Augmentation** | Random flip, crop, brightness, contrast |
| **TensorBoard** | Loss, accuracy, learning rate logged per epoch |
| **Model Checkpointing** | Saves best val-accuracy model automatically |
| **Early Stopping** | Configurable patience |
| **Evaluation** | Confusion matrix heatmap + classification report |
| **Inference** | Single image or batch folder prediction script |

---

## 📁 Project Structure

```
cnn-image-classifier/
├── models/
│   └── cnn_model.py        # CNN architecture (configurable depth/width)
├── utils/
│   ├── data_utils.py       # Data loading, normalization, tf.data pipelines
│   └── metrics.py          # Confusion matrix, training curves, predictions viz
├── train.py                # Custom training loop (main entry point)
├── predict.py              # Inference script
├── config.yaml             # Hyperparameter config
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/cnn-image-classifier.git
cd cnn-image-classifier
pip install -r requirements.txt
```

### 2. Train on CIFAR-10

```bash
python train.py
```

CIFAR-10 (~170 MB) downloads automatically on first run.

### 3. Custom hyperparameters

```bash
python train.py \
  --epochs 80 \
  --batch_size 64 \
  --lr 5e-4 \
  --filters 64 128 256 \
  --dropout 0.5 \
  --patience 15
```

### 4. Monitor with TensorBoard

```bash
tensorboard --logdir logs
```

### 5. Run inference

```bash
# Single image
python predict.py --model checkpoints/best_model.keras --image path/to/image.jpg

# Entire folder
python predict.py --model checkpoints/best_model.keras --folder path/to/images/
```

---

## 🏗️ Architecture

The default CNN has **3 convolutional stages**, each with 2 `Conv → BN → ReLU` blocks followed by max pooling:

```
Input (32×32×3)
  ↓
[Conv3x3 → BN → ReLU] × 2 → MaxPool  (32 filters)
  ↓
[Conv3x3 → BN → ReLU] × 2 → MaxPool  (64 filters)
  ↓
[Conv3x3 → BN → ReLU] × 2 → MaxPool  (128 filters)
  ↓
GlobalAveragePooling
  ↓
Dense(256) → Dropout
  ↓
Dense(10, softmax)
```

Fully configurable via `--filters` flag or `config.yaml`.

---

## 📊 Expected Results (CIFAR-10)

| Metric | Value |
|---|---|
| Test Accuracy | ~85–88% |
| Test Top-5 Accuracy | ~99%+ |

*Trained for 50 epochs with default config on a single GPU.*

---

## 📈 Outputs

After training, the `logs/` directory contains:

- `training_curves.png` — loss & accuracy over epochs
- `confusion_matrix.png` — normalized confusion matrix heatmap
- `classification_report.txt` — per-class precision / recall / F1

---

## ⚙️ Configuration

Edit `config.yaml` or pass CLI flags directly to `train.py`:

```yaml
training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 5
  patience: 10

model:
  filters: [32, 64, 128]
  dropout_rate: 0.4
```

---

## 🔧 Extending to a Custom Dataset

Replace CIFAR-10 with your own image folder (class-per-subdirectory layout):

```python
from utils.data_utils import load_from_directory

train_ds, val_ds, class_names = load_from_directory(
    data_dir="data/my_dataset",
    img_size=(64, 64),
    batch_size=32,
)
```

Then update `num_classes` in `config.yaml` to match.

---

## 📦 Requirements

- Python ≥ 3.9
- TensorFlow ≥ 2.14
- NumPy, Matplotlib, Seaborn, scikit-learn, Pillow

---

## 📄 License

MIT License — free to use, modify, and distribute.
