import numpy as np
from subpixel import SubpixelConv2D
from keras.layers import Add, Input, Conv2D, Deconv2D, MaxPool2D, UpSampling2D
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras import backend as K

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# https://github.com/titu1994/Image-Super-Resolution
# Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
######################################################
W = 128
H = 128

def createModel():
    x_input = Input((64, 64, 1))
    conv1 = Conv2D(64, (5, 5), padding='same', use_bias=True, activation='relu')(x_input)
    conv2 = Conv2D(64, (5, 5), padding='same', use_bias=True, activation='relu')(conv1)
    mp1 = MaxPool2D(pool_size=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', use_bias=True, activation='relu')(mp1)
    conv4 = Conv2D(128, (3, 3), padding='same', use_bias=True, activation='relu')(conv3)
    mp2 = MaxPool2D(pool_size=(2, 2), padding='same')(conv4)
    conv5 = Conv2D(256, (3, 3), padding='same', use_bias=True, activation='relu')(mp2)
    spc1 = SubpixelConv2D(conv5.shape, name='spc1', scale=2)(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same', use_bias=True)(spc1)
    # up1 = UpSampling2D(size=(2, 2))(conv5)
    # conv6 = Conv2D(128, (3, 3), padding='same', use_bias=True)(up1)
    conv7 = Conv2D(128, (3, 3), padding='same', use_bias=True)(conv6)
    add1 = Add()([conv4, conv7])
    spc2 = SubpixelConv2D(add1.shape, name='spc2', scale=2)(add1)
    conv8 = Conv2D(64, (3, 3), padding='same', use_bias=True)(spc2)
    # up2 = UpSampling2D(size=(2, 2))(add1)
    # conv8 = Conv2D(64, (3, 3), padding='same', use_bias=True)(up2)
    conv9 = Conv2D(64, (3, 3), padding='same', use_bias=True)(conv8)
    add2 = Add()([conv2, conv9])
    conv10 = Conv2D(4, (3, 3), padding='same', use_bias=True)(add2)
    spc3 = SubpixelConv2D(conv10.shape, name='spc3', scale=2)(conv10)

    model = Model(x_input, spc3)

    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy']) #v1-3
    model.compile(loss=rmse, optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

def loadModel(name):
    return load_model(name, custom_objects={'rmse': rmse})
    # return load_model(name)

def rmse(y_true, y_pred):
    # diff = K.square(y_pred - y_true) * 255**2
    diff = K.square(255 * (y_pred - y_true))
    return K.sum(K.sqrt(K.sum(diff, axis=(2, 1)) / (W * H)))

if __name__ == "__main__":
    model = createModel()
