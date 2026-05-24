"""
CNN Model Definitions
Supports custom CNN architectures with configurable depth and width.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, kernel_size=3, strides=1, use_bn=True, activation="relu"):
    """Reusable Conv → BN → Activation block."""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=not use_bn)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def build_cnn(
    input_shape=(32, 32, 3),
    num_classes=10,
    filters=[32, 64, 128],
    dropout_rate=0.4,
    name="CustomCNN",
):
    """
    Build a configurable CNN for image classification.

    Args:
        input_shape: Tuple (H, W, C)
        num_classes: Number of output classes
        filters: List of filter sizes per convolutional stage
        dropout_rate: Dropout rate before the classifier head
        name: Model name

    Returns:
        A compiled Keras Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = inputs

    # Convolutional stages
    for i, f in enumerate(filters):
        x = conv_block(x, f, kernel_size=3)
        x = conv_block(x, f, kernel_size=3)
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
        if i < len(filters) - 1:
            x = layers.SpatialDropout2D(0.1)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name=name)
    return model


if __name__ == "__main__":
    model = build_cnn(input_shape=(32, 32, 3), num_classes=10)
    model.summary()
