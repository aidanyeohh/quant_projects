"""Baseline architectures used for comparison in Table 2/3/4 of the paper:
CNN, RNN, LSTM and DNN. Each maps [B, seq_len, in_channels] -> [B, pred_len]
(predicted close price for the next pred_len days). TensorFlow/Keras."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def CNNForecaster(seq_len=128, pred_len=128, in_channels=6, hidden=64):
    inputs = keras.Input(shape=(seq_len, in_channels))
    x = layers.Conv1D(hidden, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(hidden, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(pred_len)(x)
    return keras.Model(inputs, outputs, name="CNNForecaster")


def RNNForecaster(seq_len=128, pred_len=128, in_channels=6, hidden=64):
    inputs = keras.Input(shape=(seq_len, in_channels))
    x = layers.SimpleRNN(hidden)(inputs)
    outputs = layers.Dense(pred_len)(x)
    return keras.Model(inputs, outputs, name="RNNForecaster")


def LSTMForecaster(seq_len=128, pred_len=128, in_channels=6, hidden=64):
    inputs = keras.Input(shape=(seq_len, in_channels))
    x = layers.LSTM(hidden)(inputs)
    outputs = layers.Dense(pred_len)(x)
    return keras.Model(inputs, outputs, name="LSTMForecaster")


def DNNForecaster(seq_len=128, pred_len=128, in_channels=6, hidden=256):
    inputs = keras.Input(shape=(seq_len, in_channels))
    x = layers.Flatten()(inputs)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden, activation="relu")(x)
    outputs = layers.Dense(pred_len)(x)
    return keras.Model(inputs, outputs, name="DNNForecaster")


BASELINES = {
    "CNN": CNNForecaster,
    "RNN": RNNForecaster,
    "LSTM": LSTMForecaster,
    "DNN": DNNForecaster,
}
