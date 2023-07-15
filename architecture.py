import tensorflow as tf
from tensorflow import keras
from keras import layers

def init_generator(window_dim, in_dim, out_dim):
    generator = keras.Sequential(
        [
            keras.Input(shape=(window_dim, in_dim)),
            layers.LSTM(32, return_sequences=True),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(128),
            layers.Dense(out_dim, activation="tanh")
        ]
    )

    return generator

def init_discriminator(window_dim, in_dim, out_dim):
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(window_dim, in_dim)),
            layers.LSTM(100),
            layers.Dense(out_dim, activation="sigmoid")   # out_dim should be 1
        ]
    )

    return discriminator