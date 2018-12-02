import os
import numpy as np
import cv2
import neural_net
import sys


def load_images(path):
    img = []
    for p in sorted(os.listdir(path)):
        im = cv2.imread(path + p, 0)
        img.append(cv2.bitwise_not(im)/255)
    return np.array(img)

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

def main():
    test_64_path = "xray/test_images_64x64/"
    train_64_path = "xray/train_images_64x64/"
    train_128_path = "xray/train_images_128x128/"

    if len(sys.argv) != 4:
        print("Usage: main.py <--train/--test> --model <model_name>")
        exit(1)
        
    if sys.argv[1] == "--train":
        nn = createModel()
        train_images_64 = load_images(train_64_path)
        train_images_128 = load_images(train_128_path)
        nn.fit(train_images_64, train_images_128, batch_size=1000, epochs=50)
        model.save(sys.argv[3])
    elif sys.argv[1] == "--test":
        nn = model.loadModel(argv[3])
        test_images_64 = load_images(test_64_path)
        test_out_128 = nn.predict(test_images_64)
        save_images("xray/test_images_128x128", test_images_64, test_out_128)

main()
