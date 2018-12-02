import numpy as np
from keras.layers import Input, Add, Conv2D, Deconv2D, UpSampling2D, BatchNormalization, LeakyReLU
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

    conv1 = Conv2D(1, (7,7), padding='valid', use_bias=True, name='conv1')(x_input)
    bnconv1 = BatchNormalization(axis=3, name='bn_conv1')(conv1)
    rlconv1 = LeakyReLU(alpha=0.3, name='rl_conv1')(bnconv1)

    deconv1 = Deconv2D(1, (7,7), use_bias=True, name='deconv1')(rlconv1)
    bndeconv1 = BatchNormalization(axis=3, name='bn_deconv1')(deconv1)
    rldeconv1 = LeakyReLU(alpha=0.3, name='rl_deconv1')(bndeconv1)

    y_output = rldeconv1

    model = Model(x_input, y_output)
    adam = Adam(lr=0.01)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

def rmse(y_true, y_pred):
    diff = K.square(y_pred - y_true)
    return K.sum(K.sqrt(K.sum(K.sum(diff, axis=1), axis=2) / (_W * _H)))

def loadModel(modelName):
    return load_model(modelName)

def saveModel(model, modelName):
    model.save(modelName)

if __name__ == "__main__":
    print('test model')
