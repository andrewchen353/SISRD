import os
import numpy as np
import cv2
import neural_net
import loss
import argparse
from keras.models import load_model

test_64_path = "xray/test_images_64x64/"
test_128_path = "xray/test_images_128x128/"
train_64_path = "xray/train_images_64x64/"
train_128_path = "xray/train_images_128x128/"
models_path = "Andrew/models/"
result_path = "xray/"

def loadImages(path, scale=0):
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

def saveImages(path, imageNames, images):
    names = sorted(os.listdir(imageNames))
    for file, i in zip(names, np.arange(len(names))):
        img = images[i]
        w, h, _ = img.shape
        img = img.reshape((w,h))
        img = img * 255
        img = img.astype(np.uint8)
        cv2.imwrite(path + file, img)
    return

def verifyTrainModelName(modelName):
    if not modelName or ".h5" in modelName or "_v" not in modelName:
        print("Invalid model name, expected format: \'<modelName>_v<#>\'")
        exit(1)
    if os.path.exists(models_path + modelName + ".h5"):
        print("Model already exists, please increase version number")
        exit(1)

def verifyTestModelName(modelName):
    if not os.path.exists(models_path + modelName + ".h5"):
        print("Model does not exist")
        exit(1)

def verifyNetwork(network):
    if network not in neural_net.lookup:
        print(network, "is not a valid model")
        exit(1)

def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def checkDir():
    if not os.path.exists(train_64_path):
        print('Woah, you\'re in the wrong directory, go to SISRD/')
        exit(1)

def train(network, model, in_path, out_path, lr, vs, batch, epochs):
    print("Creating model...")
    nn = neural_net.lookup[network](lr)
    print("Loading images...")
    scale = network == "CNNDAE" or network == "SRResNet"
    train_input = loadImages(in_path, scale)
    train_output = loadImages(out_path)
    print("Training model...")
    nn.fit(train_input, train_output, validation_split=vs, batch_size=batch, epochs=epochs)
    print("Saving model")
    nn.save(models_path + model + ".h5")
    return nn

def test(nn, in_path, out_path):
    print("Loading test images...")
    test_images_64 = loadImages(in_path)
    print("Predicting...")
    test_out_128 = nn.predict(test_images_64)
    createDir(out_path)
    print("Saving images...")
    saveImages(out_path, in_path, test_out_128)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="bool determining whether or not to train")
    parser.add_argument("--test", action="store_true", help="bool determining whether or not to test")
    parser.add_argument("--lr", type=float, default=0.001, help="network learning rate")
    parser.add_argument("--validation", type=float, default=0.1, help="validation split for training data")
    parser.add_argument("--batch", type=int, default=128, help="batch size for calculating loss")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--model", help="model to train and version number <model_name>_v<#>")
    args = parser.parse_args()
    checkDir()

    if args.model and args.train:
        verifyTrainModelName(args.model)
        network = args.model.split('_')[0]
        verifyNetwork(network)
        nn = train(network, args.model, train_64_path, train_128_path, \
                   args.lr, args.validation, args.batch, args.epochs)
        if input("Would you like to test the model? y/n: ") == 'y':
            test(nn, test_64_path, result_path + args.model + "/")
    elif args.model and args.test:
        verifyTestModelName(args.model)
        print("Loading model...")
        nn = neural_net.loadModel(models_path + args.model)
        test(nn, test_64_path, result_path + args.model + "/")
    else:
        print("Usage: main.py <--train/--test> --model <model_name>_v#")
        exit(1)

main()
