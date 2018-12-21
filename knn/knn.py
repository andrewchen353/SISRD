import numpy as np
import data_utils
import argparse
from progressbar import ProgressBar, SimpleProgress, Percentage, ETA, Bar
from os import makedirs
from os.path import exists

training_input_dir = 'xray/train_images_64x64/'
training_output_dir = 'xray/train_images_128x128/'
test_input_dir = 'xray/test_images_64x64/'
output_path = 'outputs/'

def createDir(path):
    if not exists(path):
        makedirs(path)

def rmse(y_true, y_pred):
    W = 128
    H = 128
    diff = np.square((y_pred - y_true))
    return np.sum(np.sqrt(np.sum(diff, axis=(2, 1)) / (W * H)))

def test(test_input, train_images, patches=16):
    train_input, train_output = train_images

    numImages = test_input.shape[0]
    numTrain, lr_W, lr_H = train_input.shape
    _, hr_W, hr_H = train_output.shape
    test_output = np.zeros((numImages, hr_W, hr_H))

    lrInterval = int(lr_W / patches)
    hrInterval = int(hr_W / patches)

    numZeroPatches = 0

    print('Beginning knn...')
    widgets = [SimpleProgress(), ' ', Percentage(), ' ',
               Bar(marker='-',left='[',right=']'),
               ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=numImages)
    pbar.start()

    for i in range(numImages):
        for j in range(patches):
            for k in range(patches):
                rowStart = j * lrInterval
                rowEnd = (j + 1) * lrInterval
                colStart = k * lrInterval
                colEnd = (k + 1) * lrInterval

                index = np.argmin(np.linalg.norm(train_input[:,rowStart:rowEnd,colStart:colEnd] - test_input[i,rowStart:rowEnd,colStart:colEnd], axis=(2,1)))

                rowStart = j * hrInterval
                rowEnd = (j + 1) * hrInterval
                colStart = k * hrInterval
                colEnd = (k + 1) * hrInterval

                max_patch = train_output[index,rowStart:rowEnd,colStart:colEnd]

                test_output[i,rowStart:rowEnd,colStart:colEnd] = max_patch
        pbar.update(i)
    pbar.finish()

    return test_output

def main():
    print('Loading data from: ' + training_input_dir)
    train_input = data_utils.load_data(training_input_dir)
    print('Loading data from: ' + training_output_dir)
    train_output = data_utils.load_data(training_output_dir)
    print('Loading data from: ' + test_input_dir)
    test_input = data_utils.load_data(test_input_dir)

    numTrain = train_input.shape[0]

    train_images = (train_input, train_output)
    test_output = test(test_input, train_images, patches=16)

    print('Saving images to: knnImages/')
    createDir('knnImages/')
    data_utils.save_images('knnImages/', test_input_dir, test_output)

    # Validation
    validation_input = train_input[-1000:]
    validation_output = train_output[-1000:]
    print(validation_input.shape)
    train_input = train_input[:1000]
    train_output = train_output[:1000]
    print(train_input.shape)

    train_images = (train_input, train_output)
    validation_test_output = test(validation_input, train_images, patches=16)
    score = rmse(validation_output, validation_test_output)

    print('validation rmse: ' + str(2 * score))

if __name__ == "__main__":
    main()
