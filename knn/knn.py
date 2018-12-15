import numpy as np
import data_utils
import argparse
from os import makedirs
from os.path import exists

training_input_dir = 'xray/train_images_64x64/'
training_output_dir = 'xray/train_images_128x128/'
test_input_dir = 'xray/test_images_64x64/'
output_path = 'outputs/'

def createDir(path):
    if not exists(path):
        makedirs(path)

def main():
    print('Loading data from: ' + training_input_dir)
    train_input = data_utils.load_data(training_input_dir)
    print('Loading data from: ' + training_output_dir)
    train_output = data_utils.load_data(training_output_dir)
    print('Loading data from: ' + test_input_dir)
    test_input = data_utils.load_data(test_input_dir)

    numImages = test_input.shape[0]
    _, lr_W, lr_H = train_input.shape
    _, hr_W, hr_H = train_output.shape
    test_output = np.zeros((numImages, hr_W, hr_H))

    patches = 8
    lrInterval = int(lr_W / patches)
    hrInterval = int(hr_W / patches)

    print('Beginning knn...')
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
                test_output[i,rowStart:rowEnd,colStart:colEnd] = train_output[index,rowStart:rowEnd,colStart:colEnd]
        if i % 400 == 0:
            print('On image ' + str(i))

    print('Saving images to: knnImages/')
    createDir('knnImages/')
    data_utils.save_images('knnImages/', test_input_dir, test_output)


if __name__ == "__main__":
    main()
