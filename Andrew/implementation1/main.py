import os
import numpy as np
import cv2
import neural_net
import sys


def load_images(path):
    # img = []
    # for p in sorted(os.listdir(path)):
    #     im = cv2.imread(path + p, 0)
    #     w, h = im.shape
    #     im.resize((w, h, 1))
    #     img.append(cv2.bitwise_not(im)/255)
    # return np.array(img)
    data = []
    for file in sorted(os.listdir(path)):
        img = cv2.imread(path + file, 0)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_CUBIC)
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
        img = img.astype(np.uint8)
        imwrite(path + file, img)

def main():
    test_64_path = "xray/test_images_64x64/"
    train_64_path = "xray/train_images_64x64/"
    train_128_path = "xray/train_images_128x128/"

    if len(sys.argv) != 4:
        print("Usage: main.py <--train/--test> --model <model_name>")
        exit(1)
        
    if sys.argv[1] == "--train":
        print("Creating model...")
        nn = neural_net.createModel()
        print("Loading images...")
        train_input = load_images(train_64_path)
        print(train_input.shape)
        train_output = load_images(train_128_path)
        print(train_output.shape)
        print("Training model...")
        nn.fit(train_input, train_output, batch_size=4000, epochs=50)
        print("Saving model")
        model.save(sys.argv[3])
    elif sys.argv[1] == "--test":
        nn = load_model(sys.argv[3]) # model.loadModel(argv[3])
        test_images_64 = load_images(test_64_path)
        test_out_64 = nn.predict(test_images_64)
        # test_out_128 = [cv2.resize(img, (128,128), interpolation=INTER_CUBIC) for img in test_out_64]
        save_images("xray/test_images_128x128", test_images_64, test_out_128)

main()
