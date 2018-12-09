import numpy as np
import loss
from subpixel import SubpixelConv2D
from keras.layers import Add, Subtract, Average, Input, Conv2D, Deconv2D, Lambda
from keras.layers import MaxPooling2D, UpSampling2D, PReLU, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model, load_model

######################################################
# By default keras is using TensorFlow as a backend
######################################################

######################################################
# Implementing model in
# "Medical image denoising using convolutional denoising autoencoders"
# CNNDAE
######################################################
def CNNDAE(lr):
    x_input = Input((128, 128, 1))

    conv1 = Conv2D(64, (7, 7), padding='same', activation='relu')(x_input)
    mp1   = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (5, 5), padding='same', activation='relu')(mp1)
    mp2   = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (7, 7), padding='same', activation='relu')(mp2)
    up1   = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(64, (7, 7), padding='same', activation='relu')(up1)
    up2   = UpSampling2D(size=(2, 2))(conv4)

    conv5 = Conv2D(1, (5, 5), padding='same', activation='relu')(up2)

    model = Model(x_input, conv5)

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

######################################################
# Implementing model in
# https://github.com/titu1994/Image-Super-Resolution
# Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
######################################################
def DSRCNN(lr):
    x_input = Input((64, 64, 1))

    # conv1   = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(x_input) #v1-2
    conv1   = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(x_input)
    conv2   = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(conv1)
    deconv1 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(conv2)
    
    add1    = Add()([conv2, deconv1])
    deconv2 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(add1) #v1-2
    # avg1    = Average()([conv2, deconv1])
    # deconv2 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(avg1) #v3
    
    add2  = Add()([conv1, deconv2])
    conv3 = Conv2D(4, (3, 3), padding='same', use_bias=True, activation='relu')(add2) #v1-2
    # avg2  = Average()([conv1, deconv2])
    # conv3 = Conv2D(4, (3, 3), padding='same', use_bias=True, activation='relu')(avg2) #v3
    spc1  = SubpixelConv2D(conv3.shape, scale=2)(conv3)

    model = Model(x_input, spc1)

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy']) # v1 -> lr=0.003, v2 -> lr=0.001

    return model

######################################################
# Implementing model in
# https://github.com/titu1994/Image-Super-Resolution
# Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
######################################################
def DDSRCNN(lr):
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5, 5), padding='same', use_bias=True, activation='relu')(x_input)
    conv2 = Conv2D(64, (5, 5), padding='same', use_bias=True, activation='relu')(conv1)

    mp1   = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', use_bias=True, activation='relu')(mp1)
    conv4 = Conv2D(128, (3, 3), padding='same', use_bias=True, activation='relu')(conv3)

    mp2   = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)
    conv5 = Conv2D(256, (3, 3), padding='same', use_bias=True, activation='relu')(mp2)
    spc1  = SubpixelConv2D(conv5.shape, name='spc1', scale=2)(conv5)
    conv6 = Conv2D(128, (3, 3), padding='same', use_bias=True)(spc1)
    conv7 = Conv2D(128, (3, 3), padding='same', use_bias=True)(conv6)

    add1 = Add()([conv4, conv7])

    spc2  = SubpixelConv2D(add1.shape, name='spc2', scale=2)(add1)
    conv8 = Conv2D(64, (3, 3), padding='same', use_bias=True)(spc2)
    conv9 = Conv2D(64, (3, 3), padding='same', use_bias=True)(conv8)

    add2   = Add()([conv2, conv9])
    conv10 = Conv2D(4, (3, 3), padding='same', use_bias=True)(add2)
    spc3   = SubpixelConv2D(conv10.shape, name='spc3', scale=2)(conv10)

    model = Model(x_input, spc3)

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

######################################################
# Implementing model in
# "Deep Learning for Single Image Super-Resolution:
#  A Brief Review" SRResNet - Fig.5c
# SRResNet
######################################################
def SRResNet(lr):
    # takes input of 128x128
    x_input = Input((128, 128, 1))

    conv1  = Conv2D(1, (3, 3), padding='same', activation='relu')(x_input)
    prelu1 = PReLU(alpha_initializer='zeros')(conv1)

    conv2  = Conv2D(16, (3, 3), padding='same', activation='relu')(prelu1)
    bn1    = BatchNormalization(axis=3)(conv2)
    prelu2 = PReLU(alpha_initializer='zeros')(bn1)
    conv3  = Conv2D(16, (3, 3), padding='same', activation='relu')(prelu2)
    bn2    = BatchNormalization(axis=3)(conv3)

    add1  = Add()([prelu1, bn2])
    conv4 = Conv2D(1, (3, 3), padding='same', activation='relu')(add1) # v1, 10 epochs
    # conv5 = Conv2D(1, (3, 3), padding='same', activation='relu')(conv4) # v1, 10 epochs; v2, 40 epochs
    bn3   = BatchNormalization(axis=3)(conv4) # v3, 40 epochs

    add2  = Add()([prelu1, bn3]) # v3, 40 epochs
    conv5 = Conv2D(1, (3, 3), padding='same', activation='relu')(add2) # v3, 40 epochs

    model = Model(x_input, conv5)

    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy']) #v1-3
    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy']) # v4

    return model

######################################################
# Implementing ESRCNN -> DSRCNN
######################################################
def TEST(lr):
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(64, (5, 5), padding='same', use_bias=True, activation='relu')(x_input)

    l1_conv1 = Conv2D(64, (1, 1), padding='same', use_bias=True, activation='relu')(conv1)
    l1_conv2 = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(conv1)
    l1_conv3 = Conv2D(64, (5, 5), padding='same', use_bias=True, activation='relu')(conv1)

    add1 = Add()([l1_conv1, l1_conv2, l1_conv3])

    conv2   = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(add1)
    conv3   = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(conv2)
    deconv1 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(conv3)
    
    add2    = Add()([conv3, deconv1])
    deconv2 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(add2)
    
    add3  = Add()([conv2, deconv2])
    conv3 = Conv2D(4, (3, 3), padding='same', use_bias=True, activation='relu')(add3)
    spc1  = SubpixelConv2D(conv3.shape, scale=2)(conv3)

    model = Model(x_input, spc1)

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

def TEST2(lr):
    x_input = Input((64, 64, 1))

    conv1   = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(x_input)
    conv2   = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(conv1)
    deconv1 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(conv2)
    
    add1    = Add()([conv2, deconv1])
    deconv2 = Deconv2D(64, (3, 3), padding='same', use_bias=True)(add1)
    
    add2  = Add()([conv1, deconv2])
    conv3 = Conv2D(4, (3, 3), padding='same', use_bias=True, activation='relu')(add2)
    spc1  = SubpixelConv2D(conv3.shape, name='spc1', scale=2)(conv3)

    conv1_1 = Conv2D  (64, (5, 5), padding='same', use_bias=True, activation='relu')(x_input)
    spc2  = SubpixelConv2D(conv1_1.shape, name='spc2', scale=2)(conv1_1)

    sub1  = Subtract()([spc2, spc1])
    conv4 = Conv2D(1, (1, 1), padding='same', use_bias=True, activation='relu')(sub1)

    model = Model(x_input, conv4)

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

def ResNet(lr):
    depth = 16 #v2, v1 = 4
    x_input = Input((64, 64, 1))

    conv1  = Conv2D(1, (3, 3), padding='same', use_bias=True, activation='relu')(x_input)
    # rec = PReLU(alpha_initializer='zeros')(conv1) #v1
    rec = PReLU(alpha_initializer='zeros')(conv1) #v2

    # DnCNN network
    for i in range(depth):
        rec1  = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(rec)
        rec1 = BatchNormalization(axis=3)(rec1)
        # rec = PReLU(alpha_initializer='zeros')(rec) #v1
        rec1 = LeakyReLU(alpha=0.3)(rec1) #v2
        if i%2 == 0:
            rec = Add()([rec, rec1])
        rec = Add()([rec1, rec])
    
    conv2 = Conv2D(1, (3, 3), padding='same', use_bias=True, activation='relu')(rec)
    sub = Add()([x_input, conv2])

    conv3 = Conv2D(4, (3, 3), padding='same', use_bias=True, activation='relu')(sub)
    spc1 = SubpixelConv2D(conv3.shape, scale=2)(conv3)

    # model = Model(spc1, x_input) #v1
    model = Model(x_input, spc1) #v2

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

# https://arxiv.org/pdf/1706.00552.pdf
def IDCNN(lr):
    x_input = Input((64, 64, 1))

    conv1 = Conv2D(1 , (3, 3), padding='same', use_bias=True, activation='relu')(x_input)
    loop = conv1

    for i in range(6):
        conv2 = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(loop)
        loop  = BatchNormalization(axis=3)(conv2)

    conv3  = Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu')(loop)
    div    = Lambda(lambda inputs: inputs[0]/(inputs[1] + 1e-7))([x_input, conv3])
    lrelu1 = LeakyReLU(alpha=0.3)(div)
    conv4  = Conv2D(4, (3, 3), padding='same', use_bias=True)(lrelu1)
    spc1   = SubpixelConv2D(conv3.shape, scale=2)(conv4)

    model = Model(x_input, spc1) #v2

    model.compile(loss=loss.rmse, optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model
    

def loadModel(name):
    return load_model(name, custom_objects={'rmse': loss.rmse})
    # return load_model(name)

lookup = dict()
lookup['CNNDAE']   = CNNDAE
lookup['DSRCNN']   = DSRCNN
lookup['DDSRCNN']  = DDSRCNN
lookup['SRResNet'] = SRResNet
lookup['TEST']     = TEST
lookup['ResNet']   = ResNet
lookup['TEST2']    = TEST2
lookup['IDCNN']    = IDCNN

if __name__ == "__main__":
    cnndae   = lookup['CNNDAE'](0.001)
    dsrcnn   = lookup['DSRCNN'](0.001)
    ddsrcnn  = lookup['DDSRCNN'](0.001)
    srresnet = lookup['SRResNet'](0.001)
    test     = lookup['TEST'](0.001)
    resnet   = lookup['ResNet'](0.001)
    test2    = lookup['TEST2'](0.001)
    idcnn    = lookup['IDCNN'](0.001)
    idcnn.summary()
