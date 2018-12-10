from keras import backend as K

W = 128
H = 128
img_nrows = H
img_ncols = W

def rmse(y_true, y_pred):
    # diff = K.square(y_pred - y_true) * 255**2
    diff = K.square(255 * (y_pred - y_true))
    return K.sum(K.sqrt(K.sum(diff, axis=(2, 1)) / (W * H)))

# from https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
# x is y_pred
def total_variation_loss(dc, x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.abs(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.abs(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.abs(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.abs(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return 4e-3 * (K.sum(K.sum(a, axis=[1,2,3]) + K.sum(b, axis=[1,2,3])))

def custom_loss(y_true, y_pred):
    return rmse(y_true, y_pred) + total_variation_loss(y_true, y_pred)
