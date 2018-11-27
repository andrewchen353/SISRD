import numpy as np
from keras.layers import Input, Add, Conv2D, Deconv2D, UpSampling2D, BatchNormalization, LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
import h5py

_EPSILON = K.epsilon()
_W = 160
_H = 160

######################################################
# By default keras is using TensorFlow as a backend
######################################################

def generate_model():
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (7,7), padding='same', use_bias=True, name='conv1')(x_input)
    bnconv1 = BatchNormalization(axis=3, name='bn_conv1')(conv1)
    rlconv1 = LeakyReLU(alpha=0.3, name='rl_conv1')(bnconv1)


    conv2 = Conv2D(64, (5,5), strides=(2,2), use_bias=True, name='conv2')(rlconv1)
    bnconv2 = BatchNormalization(axis=3, name='bn_conv2')(conv2)
    rlconv2 = LeakyReLU(alpha=0.3, name='rl_conv2')(bnconv2)

    conv3 = Conv2D(128, (3,3), use_bias=True, name='conv3')(rlconv2)
    bnconv3 = BatchNormalization(axis=3, name='bn_conv3')(conv3)
    rlconv3 = LeakyReLU(alpha=0.3, name='rl_conv3')(bnconv3)

    deconv1 = Deconv2D(64, (3,3), use_bias=True, name='deconv1')(rlconv3)
    bndeconv1 = BatchNormalization(axis=3, name='bn_deconv1')(deconv1)
    rldeconv1 = LeakyReLU(alpha=0.3, name='rl_deconv1')(bndeconv1)
    sbdeconv1 = Add()([rldeconv1, rlconv2])

    deconv2 = Deconv2D(64, (5,5), strides=(2,2), output_padding=(1,1), use_bias=True, name='deconv2')(sbdeconv1)
    bndeconv2 = BatchNormalization(axis=3, name='bn_deconv2')(deconv2)
    rldeconv2 = LeakyReLU(alpha=0.3, name='rl_deconv2')(bndeconv2)
    sbdeconv2 = Add()([rldeconv2, rlconv1])

    conv4 = Conv2D(1, (1,1), use_bias=True, name='conv4')(sbdeconv2)
    y_output = UpSampling2D(size=(2,2), interpolation='bilinear')(conv4)

    model = Model(x_input, y_output)
    adam = Adam(lr=0.01)
    #model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

def rmse(y_true, y_pred):
    diff = K.square(y_pred - y_true)
    return K.sum(K.sqrt(K.sum(K.sum(diff, axis=1), axis=2) / (_W * _H)))

def load_model():
    pass

def save_model(model):
    model.save('first_attempt.h5')

if __name__ == "__main__":
    print('test imports')
    generate_model()