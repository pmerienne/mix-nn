import numpy as np
import os

from mixnn.utils import mkdirs
from keras.datasets import mnist as keras_mnist
from keras.preprocessing.image import array_to_img

IMAGE_SIZE = (28, 28, 1)


def save_image(img_array, filename):
    if not os.path.exists(filename):
        reshaped = img_array.reshape(*IMAGE_SIZE)
        array_to_img(reshaped).save(filename)
    return filename


def load_data(images_directory='/tmp/mnist-digit-images'):
    mkdirs(images_directory)

    # Load Keras data
    print('Loading MNIST data')
    (X_train, y_train), (X_validation, y_validation) = keras_mnist.load_data()

    # Write images to local directory
    print('Caching MNIST images')
    X_train = np.array([
        save_image(X_train[i, ...], '%s/train_%s.jpg' % (images_directory, i))
        for i in range(X_train.shape[0])
    ])
    X_train = X_train.reshape(X_train.shape[0], 1)

    X_validation = np.array([
        save_image(X_validation[i, ...], '%s/test_%s.jpg' % (images_directory, i))
        for i in range(X_validation.shape[0])
    ])
    X_validation = X_validation.reshape(X_validation.shape[0], 1)

    features = [
        {"name": "digit_image", "type": "image", "image_size": IMAGE_SIZE},
    ]

    return features, (X_train, y_train), (X_validation, y_validation)
