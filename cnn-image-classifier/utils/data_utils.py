"""
Data utilities: loading, preprocessing, and augmentation pipelines.
Supports CIFAR-10 out of the box; extensible for custom datasets.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path


# ─── CIFAR-10 ────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar10(normalize=True):
    """
    Load CIFAR-10 and return (x_train, y_train), (x_val, y_val), (x_test, y_test).
    A 10% validation split is carved from the training set.
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Validation split
    val_size = int(len(x_train_full) * 0.1)
    x_val, y_val = x_train_full[:val_size], y_train_full[:val_size]
    x_train, y_train = x_train_full[val_size:], y_train_full[val_size:]

    if normalize:
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
        x_train = (x_train.astype(np.float32) / 255.0 - mean) / std
        x_val   = (x_val.astype(np.float32)   / 255.0 - mean) / std
        x_test  = (x_test.astype(np.float32)  / 255.0 - mean) / std

    # Flatten labels
    y_train = y_train.flatten()
    y_val   = y_val.flatten()
    y_test  = y_test.flatten()

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ─── tf.data Pipelines ───────────────────────────────────────────────────────

def augment(image, label):
    """Standard CIFAR-10 augmentations for training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random crop (pad then crop back to original size)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    return image, label


def build_dataset(x, y, batch_size=128, augment_data=False, shuffle=False, cache=True):
    """
    Build a tf.data.Dataset from numpy arrays.

    Args:
        x: Input images array
        y: Labels array
        batch_size: Batch size
        augment_data: Apply data augmentation
        shuffle: Shuffle the dataset
        cache: Cache dataset in memory

    Returns:
        tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Custom Dataset Loader ────────────────────────────────────────────────────

def load_from_directory(data_dir, img_size=(32, 32), batch_size=128, val_split=0.2):
    """
    Load a custom image classification dataset from a directory.
    Expected structure:
        data_dir/
            class_a/  image1.jpg  image2.jpg ...
            class_b/  ...

    Args:
        data_dir: Path to root data directory
        img_size: Target image size as (H, W)
        batch_size: Batch size
        val_split: Fraction used for validation

    Returns:
        train_ds, val_ds, class_names
    """
    data_dir = Path(data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )
    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names
