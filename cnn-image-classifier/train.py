"""
Custom Training Loop with:
  - Manual gradient updates via GradientTape
  - Per-epoch train/val metrics (loss, accuracy, top-5 accuracy)
  - Learning rate scheduling (cosine decay with warmup)
  - Model checkpointing & early stopping
  - TensorBoard logging
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from models.cnn_model import build_cnn
from utils.data_utils import load_cifar10, build_dataset
from utils.metrics import compute_confusion_matrix, plot_training_curves


# ─── Reproducibility ─────────────────────────────────────────────────────────

tf.random.set_seed(42)
np.random.seed(42)


# ─── Cosine Decay with Linear Warmup ─────────────────────────────────────────

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine annealing."""

    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        cosine_lr = 0.5 * self.base_lr * (
            1 + tf.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


# ─── Training Step ────────────────────────────────────────────────────────────

@tf.function
def train_step(model, optimizer, loss_fn, x_batch, y_batch, train_metrics):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
        loss += sum(model.losses)  # L2 regularization if any

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    for m in train_metrics:
        m.update_state(y_batch, logits)
    return loss


@tf.function
def val_step(model, loss_fn, x_batch, y_batch, val_metrics):
    logits = model(x_batch, training=False)
    loss = loss_fn(y_batch, logits)
    for m in val_metrics:
        m.update_state(y_batch, logits)
    return loss


# ─── Main Trainer ─────────────────────────────────────────────────────────────

def train(cfg):
    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10()
    train_ds = build_dataset(x_train, y_train, cfg.batch_size, augment_data=True, shuffle=True)
    val_ds   = build_dataset(x_val,   y_val,   cfg.batch_size)
    test_ds  = build_dataset(x_test,  y_test,  cfg.batch_size)

    steps_per_epoch = len(x_train) // cfg.batch_size
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = steps_per_epoch * cfg.warmup_epochs

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_cnn(
        input_shape=(32, 32, 3),
        num_classes=10,
        filters=cfg.filters,
        dropout_rate=cfg.dropout,
    )
    model.summary()

    # ── Optimizer & Loss ──────────────────────────────────────────────────────
    lr_schedule = WarmupCosineDecay(cfg.lr, total_steps, warmup_steps)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=cfg.weight_decay
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_acc   = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    train_top5  = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="train_top5")
    val_acc     = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")
    val_top5    = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="val_top5")
    train_metrics = [train_acc, train_top5]
    val_metrics   = [val_acc, val_top5]

    # ── Logging & Checkpoints ─────────────────────────────────────────────────
    log_dir   = Path(cfg.log_dir) / f"run_{int(time.time())}"
    ckpt_dir  = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(log_dir))

    best_val_acc   = 0.0
    patience_count = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining for {cfg.epochs} epochs  |  logs → {log_dir}\n")

    # ── Epoch Loop ────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_loss_sum, n_batches = 0.0, 0
        for x_batch, y_batch in train_ds:
            loss = train_step(model, optimizer, loss_fn, x_batch, y_batch, train_metrics)
            train_loss_sum += loss.numpy()
            n_batches += 1
        train_loss = train_loss_sum / n_batches

        # Validate
        val_loss_sum, n_val = 0.0, 0
        for x_batch, y_batch in val_ds:
            loss = val_step(model, loss_fn, x_batch, y_batch, val_metrics)
            val_loss_sum += loss.numpy()
            n_val += 1
        val_loss = val_loss_sum / n_val

        # Collect metrics
        ep_train_acc  = train_acc.result().numpy()
        ep_val_acc    = val_acc.result().numpy()
        ep_train_top5 = train_top5.result().numpy()
        ep_val_top5   = val_top5.result().numpy()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(ep_train_acc)
        history["val_acc"].append(ep_val_acc)

        # TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("loss/train", train_loss, step=epoch)
            tf.summary.scalar("loss/val",   val_loss,   step=epoch)
            tf.summary.scalar("acc/train",  ep_train_acc, step=epoch)
            tf.summary.scalar("acc/val",    ep_val_acc,   step=epoch)
            tf.summary.scalar("top5/train", ep_train_top5, step=epoch)
            tf.summary.scalar("top5/val",   ep_val_top5,   step=epoch)
            tf.summary.scalar(
                "lr", optimizer.learning_rate(optimizer.iterations).numpy(), step=epoch
            )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  acc={ep_train_acc:.4f}  top5={ep_train_top5:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={ep_val_acc:.4f}  val_top5={ep_val_top5:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Checkpoint
        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            model.save(ckpt_dir / "best_model.keras")
            print(f"  ✓ Saved best model  (val_acc={best_val_acc:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

        # Reset metrics each epoch
        for m in train_metrics + val_metrics:
            m.reset_state()

    # ── Final Evaluation ──────────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    best_model = tf.keras.models.load_model(ckpt_dir / "best_model.keras")

    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for x_batch, y_batch in test_ds:
        preds = best_model(x_batch, training=False)
        test_acc_metric.update_state(y_batch, preds)
    print(f"\nTest Accuracy: {test_acc_metric.result().numpy():.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(history, save_dir=str(log_dir))
    compute_confusion_matrix(best_model, test_ds, save_dir=str(log_dir))

    return history


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--dropout",       type=float, default=0.4)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--filters",       type=int,   nargs="+", default=[32, 64, 128])
    parser.add_argument("--log_dir",       type=str,   default="logs")
    parser.add_argument("--ckpt_dir",      type=str,   default="checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
