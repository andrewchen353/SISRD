import numpy as np
import model
import data_utils
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model')
    args = parser.parse_args()

    if args.train and args.model:
        nn = model.generate_model()
        training_input = data_utils.load_data('xray_images/train_images_64x64/')
        training_output = data_utils.load_data('xray_images/train_images_128x128/')
        nn.fit(training_input, training_output, batch_size=1000, epochs=100)
        model.saveModel(nn, args.model)
    elif args.test and args.model:
        nn = model.loadModel(args.model)
        test_input = data_utils.load_data('xray_images/test_images_64x64/')
        test_output = nn.predict(test_input)
        print(test_output.shape)
        data_utils.save_images('xray_images/test_images_128x128/', 'xray_images/test_images_64x64/', test_output)

if __name__ == "__main__":
    main()
