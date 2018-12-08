import numpy as np
import loss
from subpixel import SubpixelConv2D
from keras.layers import Add, Input, Conv2D, Deconv2D
from keras.optimizers import Adam
from keras.models import Model, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# https://github.com/titu1994/Image-Super-Resolution
# Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
######################################################

def createModel():
    x_input = Input((64, 64, 1))
    conv1 = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(x_input)
    conv2 = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(conv1)
    deconv1 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(conv2)
    add1 = Add()([conv2, deconv1])
    deconv2 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(add1)
    add2 = Add()([conv1, deconv2])
    conv3 = Conv2D(4, (3, 3), padding='same', use_bias=True, activation='relu')(add2)
    spc1 = SubpixelConv2D(conv3.shape, scale=2)(conv3)

    model = Model(x_input, spc1)

    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.compile(loss=loss.rmse, optimizer=Adam(lr=0.001), metrics=['accuracy']) # v1 -> lr=0.003, v2 -> lr=0.001

    return model

# def loadModel(name):
#     return load_model(name, custom_objects={'rmse': rmse})
#     # return load_model(name)

# def rmse(y_true, y_pred):
#     # diff = K.square(y_pred - y_true) * 255**2
#     diff = K.square(255 * (y_pred - y_true))
#     return K.sum(K.sqrt(K.sum(diff, axis=(2, 1)) / (W * H)))

if __name__ == "__main__":
    model = createModel()
