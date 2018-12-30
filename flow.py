#!/usr/bin/python3

"""My tensorflow keras playground"""

import tensorflow as tf
from tensorflow import keras

from graph import plot_training_acc

print("Running TensorFlow", tf.__version__)


def model():
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = model()

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    history = model.fit(
        train_images,
        train_labels,
        epochs=64,
        batch_size=1024,
        validation_data=(test_images, test_labels),
        callbacks=[early_stop],
    )

    plot_training_acc(history)
