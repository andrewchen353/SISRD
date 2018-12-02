import numpy as np
from cv2 import imread, imwrite, resize, INTER_CUBIC
from os import listdir

def load_data(path):
    data = []
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        img = resize(img, (128,128), interpolation=INTER_CUBIC)
        w, h = img.shape
        img.resize((w,h,1))
        img = img.astype(np.float32) / 255
        data.append(img)
    return np.array(data)

def save_images(path, imageNames, images):
    names = sorted(listdir(imageNames))
    for file, i in zip(names, np.arange(len(names))):
        img = images[i]
        w, h, _ = img.shape
        img.resize((w,h))
        img = img * 255
        img = img.astype(np.int)
        imwrite(path + file, img)

    return

if __name__ == "__main__":
    print("data_utils test")