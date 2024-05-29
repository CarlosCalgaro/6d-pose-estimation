from lib.tccnet import TccNetBuilder
from tensorflow import keras
from keras.utils import plot_model


model = TccNetBuilder().build_model()

model.summary()

# plot_model(model)