import numpy as np
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.models import Sequential, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Medical image denoising using convolutional denoising autoencoders"
######################################################

def createModel():
    # nn_input = Input((1, 64, 64))
    model = Sequential()
    
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2), (2, 2)))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(1, (2, 2), activation='relu'))

    model.add(Flatten())
    model.add(Dense(64*64, activation='softmax'))
    model.compile(loss='rmse', optimizer=Adam(lr=0.03), metrics=['accuracy'])

    return model

def loadModel(model):
    return load_model(model)

def saveModel(model, name):
    model.save(name)
