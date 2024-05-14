import tensorflow as tf
import numpy as np
import data
import os
import math
import pickle
import matplotlib.pyplot as pl
from tensorflow.keras import backend as K

class StvNet:

    def __init__(self, input_shape=(480, 640, 3), out_vectors = True, out_classes= True) -> None:
        self.input_shape = input_shape
        self.out_vectors = out_vectors
        self.out_classes = out_classes
        self.model_name = "stvNetNew"

    def build_model(self):
        inputs = tf.keras.Input(shape = self.input_shape)
        
        # Normalization layer
        normalization_layer = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        layers = self.build_conv_bn_relu(layers)

        
        pass

    def build_conv_bn_relu(self, layers):
        layers = tf.keras.layers.Conv2D(64, 7, input_shape = self.input_shape, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(layers)
        layers = tf.keras.layers.BatchNormalization()(layers)
        layers = tf.keras.layers.Activation('relu')(layers)
        return layers
