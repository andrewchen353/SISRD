import numpy as np
import loss
from subpixel import SubpixelConv2D
from keras.layers import Input, MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.models import Model, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Medical image denoising using convolutional denoising autoencoders"
######################################################

def createModel():
    x_input = Input((64, 64, 1))
    conv1 = Conv2D(64, (7, 7), padding='same', activation='relu')(x_input)
    mp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(4, (7, 7), padding='same', activation='relu')(mp2)
    spc1 = SubpixelConv2D(conv3.shape, name='spc1', scale=2)(conv3)
    conv4 = Conv2D(4, (7, 7), padding='same', activation='relu')(spc1)
    spc2 = SubpixelConv2D(conv4.shape, name='spc2', scale=2)(conv4)
    conv5 = Conv2D(1, (5, 5), padding='same', activation='relu')(spc2)

    model = Model(x_input, conv5)

    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    model.compile(loss=loss.rmse, optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = createModel()
