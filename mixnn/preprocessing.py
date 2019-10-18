import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from keras.preprocessing.image import load_img, img_to_array, array_to_img


class SafeLabelEncoder(BaseEstimator, TransformerMixin):

    def fit(self, y):
        classes = np.unique(y)
        self.class_index = {clazz: i for i, clazz in enumerate(classes)}
        self.class_inverse_index = {i: clazz for i, clazz in enumerate(classes)}
        self.fitted = True
        return self

    def transform(self, y):
        check_is_fitted(self, 'fitted')
        transformed = np.vectorize(self.get_label)(y)
        return transformed

    def inverse_transform(self, y):
        check_is_fitted(self, 'fitted')
        transformed = np.vectorize(self.get_class)(y)
        return transformed

    def get_label(self, clazz):
        return self.class_index.get(clazz, len(self.class_index))

    def get_class(self, label):
        return self.class_inverse_index.get(label, '__UNKNOWN__')

    @property
    def nb_classes(self):
        # +1 for unknown
        return len(self.class_index) + 1


class ImageEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, image_size):
        self.image_size = image_size

    def fit(self, X):
        pass

    def transform(self, X):
        transformed = np.array([
            self.preprocess_image(img_path)
            for img_path in X
        ])
        return transformed

    def inverse_transform(self, y):
        raise NotImplementedError()

    def preprocess_image(self, img_path):
        target_size = (self.image_size[0], self.image_size[1])
        color_mode = {1: "grayscale", 3: "rgb"}[self.image_size[2]]
        orig_image = load_img(img_path, target_size=target_size, color_mode=color_mode)
        orig_image = img_to_array(orig_image)
        image = np.expand_dims(orig_image, axis=0)
        image = preprocess_input(image, mode="tf")
        return np.vstack(image)

    def load_image(self, img_path):
        img_arr = self.preprocess_image(img_path)
        return array_to_img(img_arr)
