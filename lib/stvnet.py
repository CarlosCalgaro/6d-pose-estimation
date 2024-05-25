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
        inputs = tf.keras.Input(shape = self.input_shape, dtype = np.dtype('uint8'))
        
        # Normalization layer
        normalization_layer = tf.keras.layers.Lambda(lambda x: x / 255, name = 'Normalization Layer')(inputs)
        layers = self.build_conv_bn_relu(normalization_layer)
        
        layers = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(layers)

        skip = layers
        outputs = []
        outputs.append(self.build_coordinate_output_layer(layers))
        layers = self.build_conv_layer(layers, 64, 3)
        layers = self.build_conv_layer(layers, 64, 3)
        return tf.keras.Model(
            inputs = inputs,
            outputs = outputs,
            name = self.model_name)

    def build_conv_bn_relu(self, layers):
        layers = tf.keras.layers.Conv2D(64, 7, input_shape = self.input_shape, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(layers)
        layers = tf.keras.layers.BatchNormalization()(layers)
        layers = tf.keras.layers.Activation('relu')(layers)
        return layers
    
    def build_conv_layer(self, layers, number_of_filters, kernel_size, strides = 1, dilation = 1):
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = 0)
        layers = tf.keras.layers.Conv2D(number_of_filters, 
                                        kernel_size, 
                                        strides = strides, 
                                        kernel_initializer = kernel_initializer, 
                                        padding = 'same',
                                        dilation_rate = dilation)(layers)
        layers = tf.keras.layers.BatchNormalization()(layers)
        layers = tf.keras.layers.Activation('relu')(layers)
        return layers
    
    def build_coordinate_output_layer(self, layers):
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0)
        coords = tf.keras.layers.Conv2D(
            18, 
            (1, 1), 
            name = 'coordsOut',
            kernel_initializer = kernel_initializer,
            padding = 'same'
        )(layers)
        return coords