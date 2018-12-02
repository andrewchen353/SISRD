import numpy as np
import model
import data_utils

def main():
    nn = model.generate_model()
    training_input = data_utils.load_data('xray_images/train_images_64x64/')
    training_output = data_utils.load_data('xray_images/train_images_128x128/')
    nn.fit(training_input, training_output, batch_size=1000, epochs=50)
    model.save_model(nn)

if __name__ == "__main__":
    main()
