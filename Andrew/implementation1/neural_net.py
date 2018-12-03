import numpy as np
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Medical image denoising using convolutional denoising autoencoders"
######################################################

def createModel():
    model = Sequential() #input_shape=(1, 64, 64))
    
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu', input_shape=(1, 64, 64)))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(1, (2, 2), padding='same', activation='relu'))

    # model.add(Flatten())
    # model.add(Dense(64*64, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.03), metrics=['accuracy'])

    return model
