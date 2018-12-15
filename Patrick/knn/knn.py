import numpy as np
import data_utils
import argparse
from os import makedirs
from os.path import exists
from sklearn.neighbors import KNeighborsClassifier

training_input_dir = 'xray_images/train_images_64x64/'
training_output_dir = 'xray_images/train_images_128x128/'
test_input_dir = 'xray_images/test_images_64x64/'
models_path = 'Patrick/models/'
output_path = 'outputs/'

def main():
    print('Loading data from: ' + training_input_dir)
    data = data_utils.load_data_64_to_64(training_input_dir)
    print('Loading data from: ' + training_output_dir)
    labels = data_utils.load_data_128_to_64(training_output_dir)
    print(data.shape)
    print(data[0])
    print(labels.shape)
    print(labels[0])
    print('Creating KNeighbors Classifier')
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
    print('Fitting classifier')
    knn.fit(data,labels)

if __name__ == "__main__":
    main()
