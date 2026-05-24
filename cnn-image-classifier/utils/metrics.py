"""
Evaluation utilities: confusion matrix, training curves, per-class report.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_training_curves(history: dict, save_dir: str = "logs"):
    """Plot and save loss & accuracy curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2563EB")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#DC2626", linestyle="--")
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", color="#2563EB")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#DC2626", linestyle="--")
    axes[1].set_title("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training curves saved → {out}")


def compute_confusion_matrix(
    model,
    dataset: tf.data.Dataset,
    class_names=CIFAR10_CLASSES,
    save_dir: str = "logs",
):
    """Run inference on a dataset and produce a confusion matrix + class report."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_labels = [], []
    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)
        preds  = tf.argmax(logits, axis=-1).numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)
    (save_dir / "classification_report.txt").write_text(report)

    # Confusion matrix heatmap
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out = save_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {out}")


def show_predictions(model, x_samples, y_samples, class_names=CIFAR10_CLASSES, n=10):
    """Show a grid of sample predictions with ground truth vs predicted label."""
    logits = model(x_samples[:n], training=False)
    preds  = tf.argmax(logits, axis=-1).numpy()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, ax in enumerate(axes.flat):
        img = x_samples[i]
        # Undo CIFAR-10 normalization for display
        mean = np.array([0.4914, 0.4822, 0.4465])
        std  = np.array([0.2470, 0.2435, 0.2616])
        img  = np.clip(img * std + mean, 0, 1)
        ax.imshow(img)
        color = "green" if preds[i] == y_samples[i] else "red"
        ax.set_title(
            f"GT: {class_names[y_samples[i]]}\nPred: {class_names[preds[i]]}",
            color=color, fontsize=9,
        )
        ax.axis("off")
    plt.suptitle("Sample Predictions", fontsize=14)
    plt.tight_layout()
    plt.show()
