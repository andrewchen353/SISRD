import numpy as np
from keras.layers import Input, Subtract, Add, Conv2D, Deconv2D
from keras.layers import UpSampling2D, BatchNormalization, LeakyReLU, PReLU
from keras.activations import relu
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
import h5py

from subpixel import SubpixelConv2D

_EPSILON = K.epsilon()
_W = 128
_H = 128

#########################################################
# Super Resolution CNN (SRCNN)
#########################################################

def srcnn(learningRate=0.001):
    print('Creating model of architecture \'SRCNN\'')
    x_input = Input((128, 128, 1))

    conv1 = Conv2D(64, (9,9), padding='same', use_bias=True, activation='relu', name='conv1')(x_input)
    conv2 = Conv2D(32, (5,5), padding='same', use_bias=True, activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(1 , (5,5), padding='same', use_bias=True, activation='relu', name='conv3')(conv2)

    y_output = conv3

    model = Model(x_input, y_output)
    adam = Adam(lr=learningRate)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

#########################################################
# Subpixel + Super Resolution CNN (Subpixel + SRCNN)
#########################################################

def subpixelsrcnn(learningRate=0.001):
    print('Creating model of architecture \'Subpixel + SRCNN\'')
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5,5), padding='same', use_bias=True, activation='relu', name='conv1')(x_input)
    conv2 = Conv2D(32, (3,3), padding='same', use_bias=True, activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(4 , (3,3), padding='same', use_bias=True, activation='relu', name='conv3')(conv2)
    subpix = SubpixelConv2D(conv3.shape, scale=2)(conv3)

    y_output = subpix

    model = Model(x_input, y_output)
    adam = Adam(lr=learningRate)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

#########################################################
# Expanded Super Resolution CNN (ESRCNN)
#########################################################

def esrcnn(learningRate=0.001):
    print('Creating model of architecture \'ESRCNN\'')
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5,5), padding='same', use_bias=True, activation='relu', name='conv1')(x_input)

    mid1 = Conv2D(32, (5,5), padding='same', use_bias=True, activation='relu', name='mid1')(conv1)
    mid2 = Conv2D(32, (3,3), padding='same', use_bias=True, activation='relu', name='mid2')(conv1)
    mid3 = Conv2D(32, (1,1), padding='same', use_bias=True, activation='relu', name='mid3')(conv1)

    merge = Average()([mid1, mid2, mid3])
    conv3 = Conv2D(4 , (3,3), padding='same', use_bias=True, activation='relu', name='conv3')(merge)
    subpix = SubpixelConv2D(conv3.shape, scale=2)(conv3)

    y_output = subpix

    model = Model(x_input, y_output)
    adam = Adam(lr=learningRate)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

#########################################################
# Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
#########################################################

def dsrcnn(learningRate=0.001):
    print('Creating model of architecture \'DSRCNN\'')
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5,5), padding='same', use_bias=True, activation='relu', name='conv1')(x_input)
    conv2 = Conv2D(64, (3,3), padding='same', use_bias=True, activation='relu', name='conv2')(conv1)

    deconv = Deconv2D(64, (3,3), padding='same', use_bias=True, activation='relu', name='deconv')(conv2)
    add1 = Add()([deconv, conv2])

    deconv2 = Deconv2D(64, (3,3), padding='same', use_bias=True, activation='relu', name='deconv2')(add1)
    add2 = Add()([deconv2, conv1])

    conv3 = Conv2D(4 , (3,3), padding='same', use_bias=True, activation='relu', name='conv3')(add2)
    subpix = SubpixelConv2D(conv3.shape, scale=2)(conv3)
    conv4 = Conv2D(1 , (3,3), padding='same', use_bias=True, activation='relu', name='conv4')(subpix)

    y_output = conv4

    model = Model(x_input, y_output)
    adam = Adam(lr=learningRate)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

#########################################################
# Distilled ResNet Super Resolution (Distlled ResNetSR)
#########################################################

def resnet(learningRate=0.001):
    print('Creating model of architecture \'Distilled ResNetSr\'')
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(32, (1,1), padding='same', use_bias=True, name='conv1')(x_input)
    relu1 = PReLU(alpha_initializer='zeros', name='relu1')(conv1)

    conv2 = Conv2D(32, (5,5), padding='same', use_bias=True, name='conv2')(relu1)
    batch1 = BatchNormalization(axis=3, name='batch1')(conv2)
    relu2 = PReLU(alpha_initializer='zeros', name='relu2')(batch1)

    conv3 = Conv2D(32, (3,3), padding='same', use_bias=True, name='conv3')(relu2)
    batch2 = BatchNormalization(axis=3, name='batch2')(conv3)
    relu3 = PReLU(alpha_initializer='zeros', name='relu3')(batch2)

    add1 = Add()([relu3, relu1])

    add2 = Add()([add1, relu1])

    subpix = SubpixelConv2D(add2.shape, scale=2)(add2)

    conv4 = Conv2D(64, (5,5), padding='same', use_bias=True, name='conv4')(subpix)
    relu4 = PReLU(alpha_initializer='zeros', name='relu4')(conv4)

    conv5 = Conv2D(1, (3,3), padding='same', use_bias=True, name='conv5')(relu4)
    relu5 = PReLU(alpha_initializer='zeros', name='relu5')(conv5)

    y_output = relu5

    model = Model(x_input, y_output)
    adam = Adam(lr=learningRate)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

#########################################################
# test
#########################################################

def testnet(learningRate=0.001):
    print('Creating model of architecture \'testnet\'')
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5,5), padding='same', use_bias=True, activation='relu', name='conv1')(x_input)
    conv2 = Conv2D(64, (3,3), padding='same', use_bias=True, activation='relu', name='conv2')(conv1)

    deconv = Deconv2D(64, (3,3), padding='same', use_bias=True, activation='relu', name='deconv')(conv2)
    add1 = Add()([conv2, deconv])

    deconv2 = Deconv2D(64, (3,3), padding='same', use_bias=True, activation='relu', name='deconv2')(add1)
    add2 = Add()([deconv2, conv1])

    conv3 = Conv2D(32, (3,3), padding='same', use_bias=True, activation='relu', name='conv3')(add2)
    subpix = SubpixelConv2D(conv3.shape, scale=2, name='subpix1')(conv3)

    conv1_2 = Conv2D(32, (3,3), padding='same', use_bias=True, activation='relu', name='conv1_2')(x_input)
    subpix1_1 = SubpixelConv2D(conv1_2.shape, scale=2, name='subpix1_1')(conv1_2)

    add3 = Subtract()([subpix1_1, subpix])
    conv4 = Conv2D(1 , (1,1), padding='same', use_bias=True, activation='relu', name='conv4')(add3)

    y_output = conv4

    model = Model(x_input, y_output)
    adam = Adam(lr=learningRate)
    model.compile(optimizer=adam, loss=rmse, metrics=['accuracy'])
    return model

def rmse(y_true, y_pred):
    diff = K.square(255 * (y_pred - y_true))
    return K.sum(K.sqrt(K.sum(diff, axis=(2,1)) / (_W * _H)))

def loadModel(modelName):
    return load_model(modelName, custom_objects={'rmse': rmse})
    #return load_model(modelName)

def saveModel(model, modelName):
    model.save(modelName)

def modelSummary(model):
    model.summary()

# look up dictionary for functions based on a string
lookUp = dict()
lookUp['srcnn'] = srcnn
lookUp['subpixel'] = subpixelsrcnn
lookUp['esrcnn'] = esrcnn
lookUp['dsrcnn'] = dsrcnn
lookUp['resnet'] = resnet
lookUp['test'] = testnet

if __name__ == "__main__":
    print('test model')
    #lookUp['srcnn'](0.003).summary()
    #lookUp['subpixel'](0.003).summary()
    #lookUp['esrcnn'](0.003).summary()
    #lookUp['dsrcnn'](0.003).summary()
    #lookUp['resnet']().summary()
    #lookUp['test'](0.003).summary()
