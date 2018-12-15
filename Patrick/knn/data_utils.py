import numpy as np
from cv2 import imread, imwrite, resize, INTER_CUBIC
from os import listdir

def getPatch(img, x, y):
    n = []
    for i in range(-1,1):
        for j in range(-1,1):
            n.append(padImg[x+i,y+j])

def load_data_64_to_64(path):
    data = []
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        w, h = img.shape
        padImg = np.zeros((w+2,h+2))
        padImg[1:w,1:h] = img
        for x in range(w):
            for y in range(h):
                data.append(getPatch(padImg, x, y))
    return np.array(data)

def load_data_128_to_64(path):
    data = []
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        w, h = img.shape
        img = resize(img, (w/2, h/2))
        w = w / 2
        h = h / 2
        for x in range(w):
            for y in range(h):
                data.append(img[x,y])
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
    imgs = load_data('xray_images/train_images_64x64/')
    save_images('test/', 'xray_images/train_images_64x64/', imgs)