import numpy as np
from cv2 import imread, imwrite, resize, INTER_CUBIC
import cv2
from os import listdir

def getPatch(img, x, y):
    n = []
    for i in range(-1,1):
        for j in range(-1,1):
            n.append(img[x+i,y+j])
    return n

def load_data_64_to_64(path):
    data = []
    cnt = 0
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        w, h = img.shape
        padImg = np.zeros((w+2,h+2))
        padImg[1:w+1,1:h+1] = img
        for x in range(w):
            for y in range(h):
                data.append(getPatch(padImg, x, y))
        cnt += 1
        if cnt % 1000 == 0:
            print('Finished image number ' + str(cnt))
    return np.array(data)

def load_data_128_to_64(path):
    data = []
    cnt = 0
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        w, h = img.shape
        w = int(w / 2)
        h = int(h / 2)
        img = resize(img, (w, h))
        for x in range(w):
            for y in range(h):
                data.append(img[x,y])
        cnt += 1
        if cnt % 1000 == 0:
            print('Finished image number ' + str(cnt))
    return np.array(data)

def save_images(path, imageNames, images):
    names = sorted(listdir(imageNames))
    for file, i in zip(names, np.arange(len(names))):
        img = images[i]
        w, h, _ = img.shape
        img = img.reshape((w,h))
        img = img * 255
        img = img.astype(np.uint8)
        imwrite(path + file, img)
    return

if __name__ == "__main__":
    print("data_utils test")
    img = imread('xray_images/train_images_128x128/train_04000.png',0)
    img = resize(img, (64,64))
    imwrite('test2.png', img)