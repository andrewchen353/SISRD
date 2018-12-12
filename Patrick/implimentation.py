import numpy as np
import data_utils
import argparse
from os import makedirs
from os.path import exists
from keras.layers import Input, Subtract, Add, Conv2D, Deconv2D, Flatten, Reshape
from keras.layers import UpSampling2D, BatchNormalization, LeakyReLU, PReLU, Dense
from keras.activations import relu
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback
import h5py

from subpixel import SubpixelConv2D
import loss

training_input_dir = 'xray_images/train_images_64x64/'
training_output_dir = 'xray_images/train_images_128x128/'
test_input_dir = 'xray_images/test_images_64x64/'

intermediate_train_dir = 'intermediate/train_images_128x128/'
intermediate_test_dir = 'intermediate/test_images_128x128/'
models_path = 'Patrick/models/'
output_path = 'outputs/'

def model64():
    print('Creating model of architecture \'model64\'')
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5,5), padding='same', use_bias=True, name='conv1')(x_input)
    relu1 = PReLU(alpha_initializer='zeros', name='relu1')(conv1)
    conv2 = Conv2D(64, (3,3), padding='same', use_bias=True, name='conv2')(relu1)
    relu2 = PReLU(alpha_initializer='zeros', name='relu2')(conv2)

    deconv = Deconv2D(64, (3,3), padding='same', use_bias=True, name='deconv')(relu2)
    relu3 = PReLU(alpha_initializer='zeros', name='relu3')(deconv)
    add1 = Add()([relu2, relu3])

    deconv2 = Deconv2D(64, (3,3), padding='same', use_bias=True, name='deconv2')(add1)
    relu4 = PReLU(alpha_initializer='zeros', name='relu4')(deconv2)
    add2 = Add()([relu4, relu1])

    conv3 = Conv2D(32, (3,3), padding='same', use_bias=True, name='conv3')(add2)
    relu5 = PReLU(alpha_initializer='zeros', name='relu5')(conv3)
    subpix = SubpixelConv2D(relu4.shape, scale=2, name='subpix1')(relu5)

    conv1_2 = Conv2D(32, (3,3), padding='same', use_bias=True, name='conv1_2')(x_input)
    relu6 = PReLU(alpha_initializer='zeros', name='relu6')(conv1_2)
    subpix1_1 = SubpixelConv2D(conv1_2.shape, scale=2, name='subpix1_1')(relu6)

    add3 = Add()([subpix1_1, subpix])
    conv4 = Conv2D(1 , (3,3), padding='same', use_bias=True, activation='relu', name='conv4')(add3)

    y_output = conv4

    model = Model(x_input, y_output)
    adam = Adam(lr=0.003)
    model.compile(optimizer=adam, loss=loss.custom, metrics=[loss.total_variation_loss, loss.rmse])
    return model

def model128():
    print('Creating model of architecture \'model128\'')
    x_input = Input((128, 128, 1))

    conv1 = Conv2D(64, (5,5), padding='same', use_bias=True, name='conv1')(x_input)
    relu1 = PReLU(alpha_initializer='zeros', name='relu1')(conv1)
    conv2 = Conv2D(64, (3,3), padding='same', use_bias=True, name='conv2')(relu1)
    relu2 = PReLU(alpha_initializer='zeros', name='relu2')(conv2)

    deconv = Deconv2D(64, (3,3), padding='same', use_bias=True, name='deconv')(relu2)
    relu3 = PReLU(alpha_initializer='zeros', name='relu3')(deconv)
    add1 = Add()([relu2, relu3])

    deconv2 = Deconv2D(64, (3,3), padding='same', use_bias=True, name='deconv2')(add1)
    relu4 = PReLU(alpha_initializer='zeros', name='relu4')(deconv2)
    add2 = Add()([relu4, relu1])

    conv3 = Conv2D(32, (3,3), padding='same', activation='relu', use_bias=True, name='conv3')(add2)

    y_output = conv3

    model = Model(x_input, y_output)
    adam = Adam(lr=0.003)
    model.compile(optimizer=adam, loss=loss.custom, metrics=[loss.total_variation_loss, loss.rmse])
    return model

def main():
    training_input = data_utils.load_data(training_input_dir)
    training_output = data_utils.load_data(training_output_dir)
    model = model64()
    model.fit(training_input, training_output, batch_size=64, epochs=20, validation_split=0.15)
    intermidiate_train_input = model.predict(train_input)
    intermediate_test_input = model.predict(test_input)
    c = input('Want to continue? ')
    if c == 'yes':
        del model
        model = model128()
        model.fit(intermidiate_train_input, training_output, batch_size=64, epochs=20, validation_split=0.15)
        test_output = model.predict(intermediate_test_input)
        data_utils.save_images(output_path + 'twopass/', test_input_dir, test_output)

if __name__ == "__main__":
    main()