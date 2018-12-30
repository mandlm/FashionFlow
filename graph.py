"""Tensorflow graphs"""

import matplotlib.pyplot as plt

def plot_training_acc(history):
    """Plot training and validation accuracy"""
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
