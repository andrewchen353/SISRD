import numpy as np
import data_utils
import argparse
from os import makedirs
from os.path import exists

training_input_dir = 'xray_images/train_images_64x64/'
training_output_dir = 'xray_images/train_images_128x128/'
test_input_dir = 'xray_images/test_images_64x64/'
output_path = 'outputs/'

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

    patches = 1
    lrInterval = int(lr_W / patches)
    hrInterval = int(hr_W / patches)

    print('Beginning knn...')
    for i in range(numImages):
        index = np.argmax(np.linalg.norm(train_input - test_input[i,np.newaxis], axis=(2,1)))
        test_output[i] = train_output[index]
        if i % 400 == 0:
            print('On image ' + str(i))

    print('Saving images to: knnImages/')
    data_utils.save_images('knnImages/', test_input_dir, test_output)


if __name__ == "__main__":
    main()
