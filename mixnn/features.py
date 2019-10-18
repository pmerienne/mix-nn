import numpy as np

from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, concatenate, Input, BatchNormalization, Embedding, Flatten, Conv2D,  MaxPooling2D
from mixnn.preprocessing import ImageEncoder, SafeLabelEncoder
from sklearn.preprocessing import StandardScaler


class AbstractHandler(object):

    def __init__(self, features, model_parameters):
        self.features = features
        self.model_parameters = model_parameters
        self.encoders = []

    @property
    def encoders_with_feature(self):
        return zip(self.encoders, self.features)

    def fit(self, X):
        for i, feature in enumerate(self.features):
            self.log('Fitting %s' % feature)
            feature_data = X[:, i].reshape(-1, 1)
            encoder = self.create_encoder(feature)
            encoder.fit(feature_data)
            self.encoders.append(encoder)
        return self

    def log(self, msg):
        if self.model_parameters.verbose > 0:
            print(msg)

    def create_encoder(self, feature):
        raise NotImplementedError()

    def create_sub_nn(self):
        raise NotImplementedError()

    def prepare_data(self, X):
        raise NotImplementedError()


class NumericalHandler(AbstractHandler):

    def create_encoder(self, feature):
        return StandardScaler()

    def create_sub_nn(self):
        inputs = []

        if self.features:
            nb_numerical = len(self.features)
            numerical_input = Input(name='numerical_input', shape=(nb_numerical,))
            inputs.append(numerical_input)

            output = Dense(nb_numerical, kernel_initializer='normal', use_bias=False)(numerical_input)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = Dropout(self.model_parameters.dropout_rate)(output)
            outputs = [output]
        else:
            outputs = []

        return inputs, outputs

    def prepare_data(self, X):
        numerical_data = [
            encoder.transform(X[:, i].reshape(-1, 1))
            for i, (encoder, feature) in enumerate(self.encoders_with_feature)
        ]
        numerical_data = np.hstack(numerical_data)
        return {'numerical_input': numerical_data}


class CategoricalHandler(AbstractHandler):

    def create_encoder(self, feature):
        return SafeLabelEncoder()

    def create_sub_nn(self):
        inputs = []
        outputs = []

        for feature, encoder in zip(self.features, self.encoders):
            feature_name = feature['name']
            nunique = encoder.nb_classes

            input_c = Input(
                name='%s_input' % feature_name,
                shape=(1,),
                dtype='int32'
            )
            embed_c = Embedding(
                name='%s_embedding' % feature_name,
                input_dim=nunique,
                output_dim=self.model_parameters.get_embedding_size(nunique),
            )(input_c)
            embed_c = Dropout(self.model_parameters.dropout_rate)(embed_c)
            flatten_c = Flatten()(embed_c)

            inputs.append(input_c)
            outputs.append(flatten_c)

        return inputs, outputs

    def prepare_data(self, X):
        data = {}

        # TODO: refactor with "dict comprehension"
        for i, (encoder, feature) in enumerate(self.encoders_with_feature):
            data['%s_input' % feature['name']] = encoder.transform(X[:, i].reshape(-1, 1))

        return data


class ImageHandler(AbstractHandler):

    def create_encoder(self, feature):
        image_size = self.model_parameters.get_image_size(feature)
        return ImageEncoder(image_size)

    def create_sub_nn(self):
        inputs = []
        outputs = []

        for feature in self.features:
            input, output = self.create_feature_sub_nn(feature)
            inputs.append(input)
            outputs.append(output)

        return inputs, outputs

    def create_feature_sub_nn(self, feature):
        input_name = '%s_input' % feature['name']
        image_size = self.model_parameters.get_image_size(feature)
        (width, height, depth) = image_size
        cnn = self.model_parameters.get_cnn(feature)

        if cnn == "xception":
            raise NotImplementedError()
            # TODO: not tested
            base_model = Xception(weights='imagenet')
            # base_model.summary()

            for i, layer in enumerate(base_model.layers):
                layer.name = '%s_%s' % (feature['name'], layer.name)
                layer.trainable = i > 63
                # total: 126

            # Setting layer name: https://github.com/keras-team/keras/issues/6194
            base_model.layers[0].name = input_name
            input = base_model.input
            output = base_model.get_layer('avg_pool').output # avg_pool might not exists ...
            output = Flatten()(output)

            return input, output
        elif cnn == "mobilenet":
            base_model = MobileNet(
                include_top=False,
                weights='imagenet',
                input_tensor=Input(shape=image_size),
                input_shape=image_size
            )
            # base_model.summary()

            # Rename layers to avoid conflicts
            for i, layer in enumerate(base_model.layers):
                layer.name = '%s_%s' % (feature['name'], layer.name)

            # Get input and output
            base_model.layers[0].name = input_name
            input = base_model.input
            output = base_model.layers[-6].output
            output = Flatten()(output)

            # We only train the last layers (total: 88)
            for i, layer in enumerate(base_model.layers):
                layer.trainable = i > 50

            return input, output
        elif cnn == 'small':
            return self.create_standard_cnn(input_name, width, height, depth, 2)
        elif cnn == 'medium':
            return self.create_standard_cnn(input_name, width, height, depth, 3)
        elif cnn == 'large':
            return self.create_standard_cnn(input_name, width, height, depth, 4)
        else:
            raise NotImplementedError('Unsupported ImageNN: %s' % self.model_parameters.cnn)

    def create_standard_cnn(self, input_name, width, height, depth, nb_filters):
        input = Input(name=input_name, shape=(width, height, depth))

        x = input
        for i in reversed(range(nb_filters)):
            # CONV => RELU => BN => POOL
            filters_dim = width // (2**i)
            # TODO: If your input images are greater than 128×128 you may choose to use a kernel size > 3
            x = Conv2D(filters_dim, kernel_size=(3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(self.model_parameters.dropout_rate)(x)

        # Flatten the volume
        output = Flatten()(x)

        # Finally FC => RELU => BN => DROPOUT
        output = Dense(width)(output)
        output = Activation("relu")(output)
        output = BatchNormalization()(output)
        output = Dropout(self.model_parameters.dropout_rate)(output)

        return input, output

    def prepare_data(self, X):
        data = {}

        # TODO: refactor with "dict comprehension"
        for i, (encoder, feature) in enumerate(self.encoders_with_feature):
            data['%s_input' % feature['name']] = encoder.transform(X[:, i])

        return data


class FeaturesHandler(object):

    HANDLERS = {
        "numerical": NumericalHandler,
        "categorical": CategoricalHandler,
        "image": ImageHandler,
    }

    def __init__(self, features, model_parameters):
        self.features = features
        self.model_parameters = model_parameters
        self.handlers = []

    def fit(self, X):
        for type, handler_factory in self.HANDLERS.items():
            handler, index = self.create_handler(type, handler_factory)
            if handler and index:
                handler.fit(X[:, index])
                self.handlers.append((handler, index))

    def create_handler(self, type, handler_factory):
        index = [i for i, f in enumerate(self.features) if f['type'] == type]
        handler_features = [f for f in self.features if f['type'] == type]
        if handler_features:
            handler = handler_factory(handler_features, self.model_parameters)
            return handler, index
        else:
            return None, None

    def prepare_data(self, X):
        data = {}
        for handler, index in self.handlers:
            handler_raw_data = X[:, index]
            handler_data = handler.prepare_data(handler_raw_data)
            data.update(handler_data)
        return data

    def create_nn(self, output_dim):
        inputs = []
        outputs = []
        for handler, index in self.handlers:
            hanlder_inputs, hanlder_outputs = handler.create_sub_nn()
            inputs += hanlder_inputs
            outputs += hanlder_outputs

        # TODO: 8 is a magic number
        # We should do something smart to optimize feature weighting
        outputs = [self.resize_output(output, 8) for output in outputs]

        fc = concatenate(outputs, axis=-1) if len(outputs) > 1 else outputs[0]

        for i in range(self.model_parameters.fc_layers):
            previous_fc_size = K.int_shape(fc)[1]
            fc_size = (previous_fc_size + output_dim) // 2
            fc = Dense(fc_size, kernel_initializer='normal')(fc)
            fc = Activation('relu')(fc)
            fc = Dropout(self.model_parameters.dropout_rate)(fc)

        last_activation = self.model_parameters.last_activation
        output = Dense(output_dim, kernel_initializer='normal')(fc)
        output = Activation(last_activation)(output)

        model = Model(inputs=inputs, outputs=output)
        if self.model_parameters.verbose:
            model.summary()

        loss = self.model_parameters.get_loss(output_dim)
        model.compile(optimizer='adam', loss=loss, metrics=self.model_parameters.metrics)
        return model

    def resize_output(self, output, output_size):
        output = Dense(output_size)(output)
        output = Activation('relu')(output)
        output = Dropout(self.model_parameters.dropout_rate)(output)
        return output
