import numpy as np
from cv2 import imread, imwrite
from os import listdir

def load_data(path):
    data = []
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        w, h = img.shape
        img.resize((w,h,1))
        img = img.astype(np.float32) / 255
        data.append(img)
    return np.array(data)

def save_images(path,size):
    pass

if __name__ == "__main__":
    #print(load_input_data('/Users/Patrick/Downloads/xray_images/train_images_64x64/').shape)
    print(load_data('/Users/Patrick/Downloads/xray_images/train_images_128x128/').shape)