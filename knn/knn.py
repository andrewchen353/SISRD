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

                # max_norm = float('inf')
                # for l in range(patches):
                #     for m in range(patches):
                #         rs2 = l * lrInterval
                #         re2 = (l + 1) * lrInterval
                #         cs2 = m * lrInterval
                #         ce2 = (m + 1) * lrInterval

                #         rs2_hr = l * hrInterval
                #         re2_hr = (l + 1) * hrInterval
                #         cs2_hr = m * hrInterval
                #         ce2_hr = (m + 1) * hrInterval

                #         ind1 = np.argmin(np.linalg.norm(train_input[:,rs2:re2,cs2:ce2] - test_input[i,rowStart:rowEnd,colStart:colEnd], axis=(2,1)))
                #         hr_patch1 = train_output[ind1,rs2_hr:re2_hr,cs2_hr:ce2_hr]
                #         norm1 = np.linalg.norm(train_input[ind1,rs2:re2,cs2:ce2] - test_input[i,rowStart:rowEnd,colStart:colEnd])
                #         if (norm1 < max_norm):
                #             max_patch = hr_patch1
                #             max_norm = norm1

                #         # ind2 = np.argmin(np.linalg.norm(train_input[:,re2-1:rs2-1:-1,cs2:ce2] - test_input[i,rowStart:rowEnd,colStart:colEnd], axis=(2,1)))
                #         # hr_patch2 = train_output[ind1,re2_hr-1:rs2_hr-1:-1,cs2_hr:ce2_hr]
                #         # norm2 = np.linalg.norm(train_input[ind2,re2-1:rs2-1:-1,cs2:ce2] - test_input[i,rowStart:rowEnd,colStart:colEnd])
                #         # if (norm2 < max_norm):
                #         #     max_patch = hr_patch2
                #         #     max_norm = norm2

                #         # ind3 = np.argmin(np.linalg.norm(train_input[:,rs2:re2,ce2-1:cs2-1:-1] - test_input[i,rowStart:rowEnd,colStart:colEnd], axis=(2,1)))
                #         # hr_patch3 = train_output[ind1,rs2_hr:re2_hr,ce2_hr-1:cs2_hr-1:-1]
                #         # norm3 = np.linalg.norm(train_input[ind2,rs2:re2,ce2-1:cs2-1:-1] - test_input[i,rowStart:rowEnd,colStart:colEnd])
                #         # if (norm3 < max_norm):
                #         #     max_patch = hr_patch3
                #         #     max_norm = norm3

                #         # ind4 = np.argmin(np.linalg.norm(train_input[:,re2-1:rs2-1:-1,ce2-1:cs2-1:-1] - test_input[i,rowStart:rowEnd,colStart:colEnd], axis=(2,1)))
                #         # hr_patch4 = train_output[ind1,re2_hr-1:rs2_hr-1:-1,ce2_hr-1:cs2_hr-1:-1]
                #         # norm4 = np.linalg.norm(train_input[ind2,re2-1:rs2-1:-1,ce2-1:cs2-1:-1] - test_input[i,rowStart:rowEnd,colStart:colEnd])
                #         # if (norm4 < max_norm):
                #         #     max_patch = hr_patch4
                #         #     max_norm = norm4

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

    # Validation
    # validation_input = train_input[:1000]
    # validation_output = train_output[:1000]
    # print(validation_input.shape)
    # train_input = train_input[1000:]
    # train_output = train_output[1000:]
    # print(train_input.shape)

    # train_images = (train_input, train_output)
    # validation_test_output = test(validation_input, train_images, patches=16)
    # score = rmse(validation_output, validation_test_output)

    # print('validation rmse: ' + str(2 * score))

    train_images = (train_input, train_output)
    test_output = test(test_input, train_images, patches=16)

    print('Saving images to: knnImages/')
    createDir('knnImages/')
    data_utils.save_images('knnImages/', test_input_dir, test_output)


if __name__ == "__main__":
    main()
