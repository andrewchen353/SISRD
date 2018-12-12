from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import data_utils2

training_input_dir = 'xray_images/train_images_64x64/'
training_output_dir = 'xray_images/train_images_128x128/'
test_input_dir = 'xray_images/test_images_64x64/'

def main():
    training_input = data_utils2.load_data(training_input_dir)
    training_output = data_utils2.load_data(training_output_dir)
    test_input = data_utils2.load_data(test_input_dir)
    print("finish loading data")
    svr = SVR()
    r = MultiOutputRegressor(svr).fit(training_input,training_output)
    print("finish training")
    test_output = r.predict(test_input)
    data_utils2.save_data('test/',test_input_dir,test_output)
    print("done predicting")

if __name__ == "__main__":
    main()