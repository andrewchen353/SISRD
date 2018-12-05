import numpy as np
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Medical image denoising using convolutional denoising autoencoders"
######################################################

def createModel():
    # model = Sequential() #input_shape=(1, 64, 64))
    
    # model.add(Conv2D(64, (2, 2), padding='same', activation='relu', input_shape=(1, 64, 64)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))#, strides=(2, 2)))

    # model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    # model.add(UpSampling2D(size=(2, 2)))

    # model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    # model.add(UpSampling2D(size=(2, 2)))

    # model.add(Conv2D(1, (2, 2), padding='same', activation='relu'))

    x_input = Input((128, 128, 1))
    conv1 = Conv2D(64, (2, 2), padding='same', activation='relu')(x_input)
    mp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (2, 2), padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (2, 2), padding='same', activation='relu')(mp2)
    up1 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(64, (2, 2), padding='same', activation='relu')(up1)
    up2 = UpSampling2D(size=(2, 2))(conv4)
    conv5 = Conv2D(1, (2, 2), padding='same', activation='relu')(up2)

    model = Model(x_input, conv5)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

def loadModel(name):
    return load_model(name)
