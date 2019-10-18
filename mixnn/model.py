import numpy as np

from mixnn.generator import MixGenerator
from mixnn.features import FeaturesHandler
from mixnn.utils import GetBest
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from keras.models import Model
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class MixNNModel(object):

    def __init__(self,
                 features,
                 type,
                 loss,
                 metrics,
                 epochs=100,
                 batch_size=32,
                 class_weight="auto",
                 embedding_size=None,
                 max_embedding_size=3,
                 cnn="small",
                 fc_layers=1,
                 dropout_rate=0.25,
                 early_stopping=True,
                 early_stopping_rounds=5,
                 reduce_lr=True,
                 reduce_lr_patience=2,
                 reduce_lr_factor=0.5,
                 verbose=1):
        self.type = type
        self.features = features
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.embedding_size = embedding_size
        self.max_embedding_size = max_embedding_size
        self.cnn = cnn
        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.reduce_lr = reduce_lr
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.verbose = verbose

        self.features_handler = None
        self.model = None
        self.target_encoder = None

    def fit(self, X, y, validation_data=None, plot=False):
        model_parameters = ModelParameters(self)

        # Fit features handler
        self.features_handler = FeaturesHandler(self.features, model_parameters)
        self.features_handler.fit(X)

        # Fit
        self.fit_target_encoder(y)
        train_generator, validation_generator = self.create_generators(X, y, validation_data)

        # Check parameters
        class_weight = model_parameters.get_class_weight(y)
        output_dim = model_parameters.get_output_dim(y)

        self.log('Creating NN')
        self.model = self.features_handler.create_nn(output_dim)

        self.log('Fitting model')
        history = self.model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.get_callbacks(validation_data, plot),
            class_weight=class_weight,
        )
        return history

    def fit_target_encoder(self, y):
        self.log('Fitting Target Encoder')
        if self.type == 'regression':
            self.target_encoder = StandardScaler()
        elif self.type == 'classification':
            self.target_encoder = OneHotEncoder(sparse=False)
        self.target_encoder.fit(y.reshape(-1, 1))

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        transformed = self.transform(X)
        return transformed

    def transform(self, X):
        generator, _ = self.create_generators(X)
        transformer_model = self.create_transformer_model()
        transformed = transformer_model.predict_generator(generator)
        return transformed

    def get_callbacks(self, validation_data, plot):
        callbacks = []
        if self.early_stopping and validation_data is not None:
            patience = self.early_stopping_rounds
            callbacks.append(EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=patience))
            callbacks.append(GetBest(monitor='val_%s' % self.metrics[0], verbose=self.verbose))

        if plot:
            from livelossplot.keras import PlotLossesCallback
            callbacks.append(PlotLossesCallback())

        if self.reduce_lr:
            patience = self.reduce_lr_patience or self.epochs // 100
            reduce_lr = ReduceLROnPlateau(factor=self.reduce_lr_factor, patience=patience, verbose=self.verbose)
            callbacks.append(reduce_lr)

        return callbacks

    def create_generators(self, X, y=None, validation_data=None):
        train_generator = MixGenerator(
            X,
            y,
            self.features_handler,
            target_encoder=self.target_encoder,
            batch_size=self.batch_size,
        )

        validation_generator = MixGenerator(
            validation_data[0],
            validation_data[1],
            self.features_handler,
            target_encoder=self.target_encoder,
            batch_size=self.batch_size,
        ) if validation_data is not None else None

        return train_generator, validation_generator

    def predict(self, X):
        generator, _ = self.create_generators(X)
        y_pred = self.model.predict_generator(generator)
        y_pred = self.target_encoder.inverse_transform(y_pred)
        # TODO: regression [:,0]
        return y_pred

    def predict_proba(self, X):
        assert self.type == 'classification', "predict_proba is only available for classification"
        generator, _ = self.create_generators(X)
        y_pred = self.model.predict_generator(generator)
        return y_pred

    def create_transformer_model(self):
        last_layer = self.model.layers[-3]
        new_model = Model(input=self.model.input, output=last_layer.output)
        if self.verbose:
            new_model.summary()
        return new_model

    def log(self, msg):
        if self.verbose > 0:
            print(msg)

    def summary(self):
        return self.model.summary()


class MixNNRegressor(MixNNModel):

    def __init__(self,
                 features,
                 loss=None,
                 metrics=['mean_absolute_error'],
                 **kwargs):
        super(MixNNRegressor, self).__init__(features, 'regression', loss, metrics, class_weight=None, **kwargs)


class MixNNClassifier(MixNNModel):

    def __init__(self,
                 features,
                 loss=None,
                 metrics=['acc'],
                 **kwargs):
        super(MixNNClassifier, self).__init__(features, 'classification', loss, metrics, **kwargs)


class ModelParameters(object):
    def __init__(self, model):
        self.model = model

    @property
    def verbose(self):
        return self.model.verbose

    @property
    def dropout_rate(self):
        return self.model.dropout_rate

    @property
    def max_embedding_size(self):
        return self.model.max_embedding_size

    @property
    def fc_layers(self):
        return self.model.fc_layers

    @property
    def loss(self):
        return self.model.loss

    @property
    def metrics(self):
        return self.model.metrics

    @property
    def last_activation(self):
        return {"classification": 'softmax', "regression": "tanh"}[self.model.type]

    def get_output_dim(self, y):
        if self.model.type == "regression":
            # TODO: Multivariate regression
            return 1
        else:
            return len(np.unique(y))

    def get_loss(self, output_dim):
        if self.model.loss:
            loss = self.model.loss
        elif self.model.type == "regression":
            loss = "mse"
        elif self.model.type == "classification" and output_dim == 1:
            loss = "binary_crossentropy"
        elif self.model.type == "classification" and output_dim > 1:
            loss = "categorical_crossentropy"
        return loss

    def get_class_weight(self, y):
        if self.model.class_weight == 'auto':
            self.log('Computing class weight')
            counts = self.model.target_encoder.transform(y.reshape(-1, 1)).sum(axis=0)
            weights = 1.0 / counts
            weights = weights / min(weights)
            weights = dict(enumerate(weights))
            self.log("Class Weight: %s" % weights)
            return weights
        elif isinstance(self.model.class_weight, dict):
            return self.model.class_weight
        else:
            return None

    def get_embedding_size(self, nb_classes):
        if self.model.embedding_size is None:
            embedding_size = nb_classes // 2
        elif callable(self.model.embedding_size):
            embedding_size = self.model.embedding_size(nb_classes)
        elif isinstance(self.model.embedding_size, int):
            embedding_size = self.model.embedding_size
        else:
            raise RuntimeError("Unsupported embedding_size: %s" % self.model.embedding_size)

        return min(self.model.max_embedding_size, embedding_size)

    def get_cnn(self, feature):
        return feature.get("cnn") or self.model.cnn

    def get_image_size(self, feature):
        cnn = self.get_cnn(feature)
        if cnn == "xception":
            return 299, 299, 3
        elif cnn == "mobilenet":
            return 224, 224, 3
        elif cnn == "small":
            return feature.get('image_size') or (32, 32, 1)
        elif cnn == "medium":
            return feature.get('image_size') or (64, 64, 1)
        elif cnn == "large":
            return feature.get('image_size') or (128, 128, 1)

    def log(self, msg):
        if self.verbose > 0:
            print(msg)
