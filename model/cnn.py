import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.utils import np_utils

class CNN():

    def __init__(self):
        

    def cnn(self):
        model_cnn = Sequential()
        model_cnn.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 100, 1)))
        model_cnn.add(keras.layers.MaxPooling2D((2,2)))
        model_cnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model_cnn.add(keras.layers.MaxPooling2D((2, 2)))
        model_cnn.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model_cnn.add(keras.layers.Flatten())
        model_cnn.add(keras.layers.Dense(64, activation='relu'))
        model_cnn.add(keras.layers.Dense(2, activation='softmax'))
        model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model_cnn