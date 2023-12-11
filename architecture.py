import tensorflow as tf
from tensorflow import keras
from keras import layers

def init_generator(in_dim, out_dim):
    nr_units_dense = 1
    for i in out_dim:
        nr_units_dense *= i

    generator = keras.Sequential(
        [
            keras.Input(shape=in_dim),
            layers.LSTM(32, return_sequences=True),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(128),
            layers.Dense(nr_units_dense, activation="tanh"),
            layers.Reshape(out_dim)
        ]
    )

    return generator

def init_discriminator(in_dim, out_dim):
    discriminator = keras.Sequential(
        [
            keras.Input(shape=in_dim),
            layers.LSTM(100),
            layers.Dense(out_dim, activation="sigmoid")   # out_dim should be (1,)
        ]
    )

    return discriminator

def init_encoder(in_dim, out_dim):
    '''
    This neural network (encoder) is used only in
    our updated model of TAnoGAN, which we call ModifiedTAnoGAN
    '''
    nr_units_dense = 1
    for i in out_dim:
        nr_units_dense *= i

    encoder = keras.Sequential(
        [
            keras.Input(shape=in_dim),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(nr_units_dense, activation="tanh"),
            layers.Reshape(out_dim)
        ]
    )

    return encoder