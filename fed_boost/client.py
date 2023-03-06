from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    Activation,
)
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.datasets.cifar10 as cf
import numpy as np
import tensorflow as tf

from fed_boost.parameters import (
    num_filters,
    filter_size,
    pool_size,
    input_image_shape,
    output_class_size,
)
from fed_boost.data_extractor import alphas, get_split_data


class Client:
    def __init__(self, client_number, client_epochs, alpha):
        self.client_number = client_number
        self.client_epochs = client_epochs
        self.alpha = alpha
        self._set_data()

        self.conv = Conv2D(
            num_filters,
            filter_size,
            input_shape=input_image_shape,
            # strides=2,
            # padding='same',
            # activation='relu'
        )
        self.max_pool = MaxPooling2D(pool_size=pool_size)
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.dense = Dense(output_class_size, activation=None)
        self.softmax = Activation("softmax")
        self.model = Sequential(
            [
                self.conv,
                self.max_pool,
                self.dropout,
                self.flatten,
                self.dense,
                self.softmax,
            ]
        )
        self.model.compile(
            "adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # doesn't need softmax
        self.model_without_softmax = Sequential(
            [self.conv, self.max_pool, self.dropout, self.flatten, self.dense]
        )

    def _set_data(self):
        (train_images, train_labels), (
            self.test_images,
            self.test_labels,
        ) = cf.load_data()
        data_split = get_split_data(self.alpha)
        self.train_images = data_split[self.client_number]["x_train"]
        self.train_labels = data_split[self.client_number]["y_train"]
        # Normalize the images.
        self.train_images = (self.train_images / 255) - 0.5
        self.test_images = (self.test_images / 255) - 0.5

    def divide_weights(self, n):
        newweights = []
        for weight in self.model.get_weights():
            newweights.append(weight / n)
        return newweights

    def add_weights(self, weights1, weights2):
        if weights2 is None:
            return weights1
        if weights1 is None:
            return weights2
        else:
            sum = []
            for i in range(len(weights1)):
                sum.append(weights1[i] + weights2[i])
            return sum

    def train_model(self):
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor="accuracy")]
        self.model.fit(
            self.train_images,
            to_categorical(self.train_labels, output_class_size),
            epochs=self.client_epochs,
            validation_data=(
                self.train_images,
                to_categorical(self.train_labels, output_class_size),
            ),
            callbacks=callbacks,
        )

    def load_model(self, path):
        self.model_without_softmax.load_weights(self.get_save_model_path(path))
        self.model.load_weights(self.get_save_model_path(path))

    def save_model(self, path):
        self.model.save_weights(self.get_save_model_path(path), save_format='h5')
        # self.model.save(self.get_save_model_path(path))

    def get_save_model_path(self, path):
        return f"{path}/client_model_{str(self.client_number)}_al{self.alpha}.h5"

    def predict(self, x):
        if x.shape == input_image_shape:
            return np.argmax(self.model.predict(np.array([x])), axis=1)[0]
        else:
            return np.argmax(self.model.predict(x), axis=1)

    def predict_without_softmax(self, x):
        model = self.model_without_softmax
        if x.shape == input_image_shape:
            return model.predict(np.array([x]))[0]
        else:
            return model.predict(x)
