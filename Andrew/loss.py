from keras import backend as K

W = 128
H = 128

def rmse(y_true, y_pred):
    # diff = K.square(y_pred - y_true) * 255**2
    diff = K.square(255 * (y_pred - y_true))
    return K.sum(K.sqrt(K.sum(diff, axis=(2, 1)) / (W * H)))