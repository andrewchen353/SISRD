import numpy as np
from keras.layers import Add, Input, Conv2D, PReLU
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Deep Learning for Single Image Super-Resolution:
#  A Brief Review" Fig.5c
######################################################

def createModel():
    x_input = Input((128, 128, 1))
    conv1 = Conv2D(1, (3, 3), padding='same', activation='relu')(x_input)
    prelu1 = PReLU(alpha_initializer='zeros')(conv1)
    conv2 = Conv2D(16, (3, 3), padding='same', activation='relu')(prelu1)
    bn1 = BatchNormalization(axis=3)(conv2)
    prelu2 = PReLU(alpha_initializer='zeros')(bn1)
    conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')(prelu2)
    bn2 = BatchNormalization(axis=3)(conv3)
    add1 = Add()([prelu1, bn2])
    conv4 = Conv2D(1, (3, 3), padding='same', activation='relu')(add1)
    conv5 = Conv2D(1, (3, 3), padding='same', activation='relu')(conv4)

    model = Model(x_input, conv5)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

def loadModel(name):
    return load_model(name)
