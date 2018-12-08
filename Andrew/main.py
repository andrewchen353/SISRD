import os
import numpy as np
import cv2
import CNNDAE, DDSRCNN, DSRCNN, SRREesNet
import loss
import argparse
from keras.models import load_model

def load_images(path, scale=0):
    data = []
    for file in sorted(os.listdir(path)):
        img = cv2.imread(path + file, 0)
        if scale:
            img = cv2.resize(img, (128,128), interpolation=cv2.INTER_CUBIC)
        w, h = img.shape
        img = img.reshape((w,h,1))
        img = img.astype(np.float32) / 255
        data.append(img)
    return np.array(data)

def save_images(path, imageNames, images):
    names = sorted(os.listdir(imageNames))
    for file, i in zip(names, np.arange(len(names))):
        img = images[i]
        w, h, _ = img.shape
        img = img.reshape((w,h))
        img = img * 255
        img = img.astype(np.uint8)
        cv2.imwrite(path + file, img)
    return

def loadModel(name):
    return load_model(name, custom_objects={'rmse': loss.rmse})
    # return load_model(name)

def main():
    test_64_path = "xray/test_images_64x64/"
    test_128_path = "xray/test_images_128x128/"
    train_64_path = "xray/train_images_64x64/"
    train_128_path = "xray/train_images_128x128/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--network")
    parser.add_argument("--model")
    args = parser.parse_args()

    if args.model:
        if args.train and args.network:
            print("Creating model...")
            if args.network == "CNNDAE":
                nn = CNNDAE.createModel()
            elif args.network == "DDSRCNN":
                nn = DDSRCNN.createModel()
            elif args.network == "DSRCNN":
                nn = DSRCNN.createModel()
            elif args.network == "SRResNet":
                nn = SRResNet.createModel()
            else:
                print(args.network, "is not a valid model")
                exit(1)
            print("Loading images...")
            scale = args.network == "CNNDAE" or args.network == "SRResNet"
            train_input = load_images(train_64_path, scale)
            print(train_input.shape)
            train_output = load_images(train_128_path)
            print(train_output.shape)
            print("Training model...")
            nn.fit(train_input, train_output, validation_split=0.1, batch_size=128, epochs=20)
            print("Saving model")
            nn.save(args.model)
        if args.test:
            if not args.train:
                print("Loading model...")
                nn = loadModel(args.model)
            print("Loading test images...")
            test_images_64 = load_images(test_64_path)
            print("Predicting...")
            test_out_128 = nn.predict(test_images_64)
            print(test_out_128.shape)
            if not os.path.exists(test_128_path):
                os.makedirs(test_128_path)
            print("Saving images...")
            save_images(test_128_path, test_64_path, test_out_128)
        else:
            print("Usage: main.py <--train/--test> --model <model_name>")
            exit(1)
    else:
        print("Usage: main.py <--train/--test> --model <model_name>")
        exit(1)

main()
