import math

from keras.utils import Sequence


class MixGenerator(Sequence):
    def __init__(self, X, y,
                 features_handler,
                 target_encoder=None,
                 batch_size=32):
        self.X = X
        self.y = y
        self.features_handler = features_handler
        self.target_encoder = target_encoder
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        start_index = idx * self.batch_size
        end_index = (idx + 1) * self.batch_size
        batch = self.X[start_index:end_index, ...]

        encoded_data = self.features_handler.prepare_data(batch)
        if self.y is None:
            return encoded_data, None
        else:
            batch_target = self.y[start_index:end_index, ...]
            encoded_target = self.prepare_target(batch_target)
            return encoded_data, encoded_target

    def prepare_target(self, y):
        y = self.target_encoder.transform(y.reshape(-1, 1))
        return y

# TODO: Image generator should transform training images:
# x = self.image_data_generator.apply_transform(x.astype(self.dtype), params)
# x = self.image_data_generator.standardize(x)
