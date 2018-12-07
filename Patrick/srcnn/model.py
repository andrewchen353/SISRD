import numpy as np
from keras.layers import Input, Add, Conv2D, Deconv2D, UpSampling2D, BatchNormalization, LeakyReLU
from keras.activations import relu
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
import h5py

_EPSILON = K.epsilon()
_W = 128
_H = 128

######################################################
# By default keras is using TensorFlow as a backend
######################################################

def generate_model():
    x_input = Input((128, 128, 1))

    conv1 = Conv2D(64, (9,9), padding='same', use_bias=True, activation='relu', name='conv1')(x_input)
    conv2 = Conv2D(32, (5,5), padding='same', use_bias=True, activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(1 , (5,5), padding='same', use_bias=True, activation='relu', name='conv3')(conv2)

    y_output = conv3

    model = Model(x_input, y_output)
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

def rmse(y_true, y_pred):
    diff = K.square(y_pred - y_true)
    return K.sum(K.sqrt(K.sum(diff, axis=(2,1)) / (_W * _H)))

def loadModel(modelName):
    return load_model(modelName, custom_objects={'rmse': rmse})
    #return load_model(modelName)

def saveModel(model, modelName):
    model.save(modelName)

if __name__ == "__main__":
    print('test model')
