import numpy as np
import loss
from keras.layers import Add, Input, Conv2D, PReLU, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import backend as K

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Deep Learning for Single Image Super-Resolution:
#  A Brief Review" SRResNet - Fig.5c
######################################################

def createModel():
    # takes input of 128x128
    x_input = Input((128, 128, 1))
    conv1 = Conv2D(1, (3, 3), padding='same', activation='relu')(x_input)
    prelu1 = PReLU(alpha_initializer='zeros')(conv1)
    conv2 = Conv2D(16, (3, 3), padding='same', activation='relu')(prelu1)
    bn1 = BatchNormalization(axis=3)(conv2)
    prelu2 = PReLU(alpha_initializer='zeros')(bn1)
    conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')(prelu2)
    bn2 = BatchNormalization(axis=3)(conv3)
    add1 = Add()([prelu1, bn2])
    conv4 = Conv2D(1, (3, 3), padding='same', activation='relu')(add1) # v1, 10 epochs
    # conv5 = Conv2D(1, (3, 3), padding='same', activation='relu')(conv4) # v1, 10 epochs; v2, 40 epochs
    bn3 = BatchNormalization(axis=3)(conv4) # v3, 40 epochs
    add2 = Add()([prelu1, bn3]) # v3, 40 epochs
    conv5 = Conv2D(1, (3, 3), padding='same', activation='relu')(add2) # v3, 40 epochs

    model = Model(x_input, conv5)

    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy']) #v1-3
    model.compile(loss=loss.rmse, optimizer=Adam(lr=0.001), metrics=['accuracy']) # v4

    return model

if __name__ == "__main__":
    model = createModel()
