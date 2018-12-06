import numpy as np
import model
import data_utils
import argparse

training_input_dir = 'xray_images/train_images_64x64/'
training_output_dir = 'xray_images/train_images_128x128/'
test_input_dir = 'xray_images/test_images_64x64/'
test_output_dir = 'xray_images/test_images_128x128/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model')
    args = parser.parse_args()

    if args.train and args.model:
        nn = model.generate_model()
        training_input = data_utils.load_data(training_input_dir)
        training_output = data_utils.load_data(training_output_dir)
        nn.fit(training_input, training_output, batch_size=128, epochs=30)
        model.saveModel(nn, args.model)
        test = input("Do you want to test with the test images too? ")
        if test == 'yes':
            test_input = data_utils.load_data(test_input_dir)
            test_output = nn.predict(test_input)
            print(test_output.shape)
            data_utils.save_images(test_output_dir, test_input_dir, test_output)
    elif args.test and args.model:
        nn = model.loadModel(args.model)
        test_input = data_utils.load_data(test_input_dir)
        test_output = nn.predict(test_input)
        print(test_output.shape)
        data_utils.save_images(test_output_dir, test_input_dir, test_output)

if __name__ == "__main__":
    main()
