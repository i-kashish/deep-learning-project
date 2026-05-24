"""
Train a CNN on MNIST and save the model.
Run this ONCE to generate: model/mnist_cnn.keras
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

tf.random.set_seed(42)

# ── Load MNIST ────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., np.newaxis].astype("float32") / 255.0
x_test  = x_test[..., np.newaxis].astype("float32") / 255.0

# Augmentation
data_aug = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# ── Model ─────────────────────────────────────────────────────────────────────
inputs = tf.keras.Input(shape=(28, 28, 1))
x = data_aug(inputs)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = Model(inputs, outputs, name="MnistCNN")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Train ─────────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
]

model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
)

loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {acc:.4f}")

Path("model").mkdir(exist_ok=True)
model.save("model/mnist_cnn.keras")
print("Model saved → model/mnist_cnn.keras")
