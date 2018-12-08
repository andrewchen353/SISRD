import numpy as np
import model
import data_utils
import argparse
from os import makedirs
from os.path import exists

training_input_dir = 'xray_images/train_images_64x64/'
training_output_dir = 'xray_images/train_images_128x128/'
test_input_dir = 'xray_images/test_images_64x64/'
# test_output_dir = 'xray_images/test_images_128x128/'
models_path = 'Patrick/models/'
output_path = 'outputs/'

def train(modelName, epochs, batch, lr, validation_split):
    key = modelName.split('_')[0] 
    if key not in model.lookUp:
        print('Invalid model given')
        exit(1)
    if exists(models_path + modelName + '.h5'):
        print('This model has already been created, increase version number')
        exit(1)
    nn = model.lookUp[key](lr)
    print('Loading training input images from: ' + training_input_dir)
    training_input = data_utils.load_data(training_input_dir)
    print('Loading training output images from: ' + training_output_dir)
    training_output = data_utils.load_data(training_output_dir)
    print('Beginning training...')
    nn.fit(training_input, training_output, batch_size=batch, epochs=epochs, validation_split=validation_split)
    print('Saving model to: ' + models_path + modelName + '.h5')
    model.saveModel(nn, models_path + modelName + '.h5')
    return nn

def test(nn, storePath):
    print('Loading test input images: ' + test_input_dir)
    test_input = data_utils.load_data(test_input_dir)
    print('Beginning to predict...')
    test_output = nn.predict(test_input)
    if not exists(output_path + storePath):
        makedirs(output_path + storePath)
    print('Saving test output images to: ' + output_path + storePath)
    data_utils.save_images(output_path + storePath, test_input_dir, test_output)

def checkValid(modelName):
    # expected format: srcnn_v2
    if modelName == None or '.h5' in modelName or '_v' not in modelName:
        print('Invalid model name, expected format: \'srcnn_v2\'')
        exit(1)

def checkCurrDirectory():
    if not exists(training_input_dir):
        print('Woah, you\'re in the wrong directory, go to SISRD/')
        exit(1)

def main():
    parser = argparse.ArgumentParser(description='This code will train and test with a new model or loaded one given the desired type of architecture')
    parser.add_argument('--train', action='store_true', help='bool determining whether to train or not')
    parser.add_argument('--test', action='store_true', help='bool determining whether to test or not')
    parser.add_argument('--model', help='model name, to be of the form \'srcnn_v2\' with model type and version')
    parser.add_argument('--epochs', help='number of epochs to train with', type=int, default=20)
    parser.add_argument('--batch', help='batch size to calculate loss', type=int, default=64)
    parser.add_argument('--lr', help='model learning rate', type=float, default=0.001)
    parser.add_argument('--validation', help='validation split for training data', type=float, default=0.1)
    args = parser.parse_args()
    checkValid(args.model)
    checkCurrDirectory()

    if args.train and args.model:
        nn = train(args.model, args.epochs, args.batch, args.lr, args.validation)
        test_or_not = input("Do you want to test with the test images too? ")
        if test_or_not == 'yes':
            test(nn, args.model + '/')
    elif args.test and args.model:
        nn = model.loadModel(models_path + args.model + '.h5')
        test(nn, args.model + '/')

if __name__ == "__main__":
#    print("srcnn_v2".split('_')[0])
#    print(models_path + 'srcnn_v2' + '.h5')
#    print(output_path + 'srcnn_v2' + '/')
    main()
