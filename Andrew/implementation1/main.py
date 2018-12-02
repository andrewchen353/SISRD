import os
import numpy as np
import cv2
import neural_net
from PIL import Image


def load_images(path):
    img = []
    for p in sorted(os.listdir(path)):
        im = cv2.imread(path + p, 0)
        img.append(cv2.bitwise_not(im)/255)
    return np.array(img)

def main():
    train_64_path = "train_images_64x64/"
    train_128_path = "train_images_128x128/"
    images_64 = load_images(train_64_path)
    images_128 = load_images(train_128_path)

main()
