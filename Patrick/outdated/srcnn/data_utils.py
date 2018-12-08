import numpy as np
from cv2 import imread, imwrite, resize, INTER_CUBIC, imshow, waitKey, destroyAllWindows
from os import listdir

def load_data(path):
    data = []
    print(path)
    for file in sorted(listdir(path)):
        img = imread(path + file, 0)
        if img.shape[0] != 128:
            img = resize(img, (128,128), interpolation=INTER_CUBIC)
        w, h = img.shape
        img = img.reshape((w,h,1))
        img = img.astype(np.float32) / 255
        data.append(img)
    print(len(data))
    return np.array(data)

def save_images(path, imageNames, images):
    names = sorted(listdir(imageNames))
    for file, i in zip(names, np.arange(len(names))):
        img = images[i]
        w, h, _ = img.shape
        img = img.reshape((w,h))
        img = img * 255
        img = img.astype(np.uint8)
        imwrite(path + file, img)
    return

if __name__ == "__main__":
    print("data_utils test")
    imgs = load_data('xray_images/train_images_64x64/')
    save_images('test/', 'xray_images/train_images_64x64/', imgs)
