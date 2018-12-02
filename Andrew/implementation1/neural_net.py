import numpy as np
from keras.layers import Input, MaxPooling2D, UpSampling2D

def nn():
    nn_input = Input((64, 64, 1))