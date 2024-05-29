import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.initializers import GlorotUniform
import numpy as np

class TccNetBuilder():

  def __init__(self, input_shape = (480, 640, 3)):
    print("IT LIVES!")
    self.__input_shape = input_shape
    self.__model_name = 'TccNetBuilder'
    pass


  def build_model(self):
    x_in = keras.Input(self.__input_shape, dtype= np.dtype('uint8'))
    x = layers.Lambda(lambda x: x / 255)(x_in)
    x = layers.Conv2D(64, 7, input_shape = self.__input_shape, kernel_initializer=GlorotUniform(seed=0), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    res1 = x
    x = layers.MaxPool2D(pool_size= 3, strides= 2, padding= 'same')(x)
    x = self.build_res_block(x, 64)
    x = self.build_res_block(x, 64)
    
    res2 = x

    x = self.build_res_block(x, 128, kernel_size= 3, down_sample= True)
    x = self.build_res_block(x, 128)

    x = self.build_res_block(x, 256, kernel_size= 3, down_sample= True)
    x = self.build_res_block(x, 256)

    # Perdemos um dilation aqui 
    x = self.build_res_block(x, 512, kernel_size= 3, down_sample = True)
    x = self.build_res_block(x, 512, kernel_size= 3, dilation= 2)


    x = self.build_res_block(x, 256, kernel_size= 3, down_sample= True)
    x = self.build_res_block(x, 256)


    x = self.build_res_block(x, 128, kernel_size= 3, down_sample= True)


    x = self.build_res_block(x, 64, kernel_size= 3, down_sample= True)

    x = self.build_res_block(x, 32, kernel_size= 3, down_sample= True)

    return keras.Model(inputs = x_in,
                       outputs = self.build_outputs(x),
                       name = self.__model_name)
  def build_outputs(self, x):
    outputs = []
    outputs.append(
      layers.Conv2D(18, (1,1), kernel_initializer = GlorotUniform(seed=0), padding = 'same') (x)
    )
	
    return outputs

  def build_res_block(self, x, channels, kernel_size = 3, dilation = 1, down_sample = False):
    skip = x
    strides = [2, 1] if down_sample else [1, 1]
    fx = layers.Conv2D(channels, kernel_size, strides= strides[0], dilation_rate= dilation, padding='same', kernel_initializer= GlorotUniform(seed=0))(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation('relu')(fx)

    fx = layers.Conv2D(channels, kernel_size, strides= strides[1], dilation_rate= dilation, padding='same', kernel_initializer= GlorotUniform(seed=0))(fx)
    fx = layers.BatchNormalization()(fx)
    # fx = layers.Activation('relu')(fx)

    if down_sample:
      old = skip.shape
      skip = layers.Conv2D(channels, kernel_size=(1, 1), strides= 2, kernel_initializer= GlorotUniform(seed= 0))(skip)
      skip = layers.BatchNormalization()(skip)
      print("Reshaping skip from: ", old, " to: ", skip.shape)
    print("Shapes: ", fx.shape, " ", skip.shape)
    fx = layers.Add()([fx, skip])

    out = layers.Activation('relu')(fx)
    return out