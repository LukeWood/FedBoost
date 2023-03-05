from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.datasets.cifar10 as cf
import numpy as np
import tensorflow as tf
import sys

from fed_boost.parameters import (
    num_filters,
    filter_size,
    pool_size,
    input_image_shape,
    output_class_size,
    client_size,
    client_epochs,
)
from fed_boost.data_extractor import alphas, get_split_data


class Client:
    def __init__(self, client_number, client_epochs, alpha):
        self.client_number = client_number
        self.client_epochs = client_epochs
        self.alpha = alpha
        self._set_data()
        self._set_model()

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

    def _set_model(self):
        self.model = Sequential(
            [
                Conv2D(
                    num_filters,
                    filter_size,
                    input_shape=input_image_shape,
                    # strides=2,
                    # padding='same',
                    # activation='relu'
                ),
                MaxPooling2D(pool_size=pool_size),
                Dropout(0.5),
                Flatten(),
                Dense(output_class_size, activation="softmax"),
            ]
        )

        self.model.compile(
            "adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_model_without_softmax(self):
        model = Sequential(
            [
                Conv2D(
                    num_filters,
                    filter_size,
                    input_shape=input_image_shape,
                    # strides=2,
                    # padding='same',
                    # activation='relu'
                ),
                MaxPooling2D(pool_size=pool_size),
                Dropout(0.5),
                Flatten(),
                Dense(output_class_size),
            ]
        )

        model.set_weights(self.model.get_weights())
        return model

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
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3)
        ]
        self.model.fit(
            self.train_images,
            to_categorical(self.train_labels, output_class_size),
            epochs=self.client_epochs,
            validation_data=(
                self.train_images,
                to_categorical(self.train_labels, output_class_size),
            ),
            callbacks=callbacks
        )

    def load_model(self, path):
        self.model.load_weights(self.get_save_model_path())

    def save_model(self, path):
        self.model.save(self.get_save_model_path(path))

    def get_save_model_path(self, path):
        return f"{path}/client_model_{str(self.client_number)}_al{self.alpha}.h5"

    def predict(self, x):
        if x.shape == input_image_shape:
            return np.argmax(self.model.predict(np.array([x])), axis=1)[0]
        else:
            return np.argmax(self.model.predict(x), axis=1)

    def predict_without_softmax(self, x):
        model = self.get_model_without_softmax()
        if x.shape == input_image_shape:
            return model.predict(np.array([x]))[0]
        else:
            return model.predict(x)


# client = Client(10,100,'0.00')
# client.load_model(client.model_path)
# print(client.predict(client.test_images[1]))
# print(client.predict_without_softmax(client.test_images[1]).shape)
# newweights = []
# for weight in client.model.get_weights():
#   newweights.append(weight/2)
# client.model.set_weights(newweights)
# client.divide_weights(2)
# client.add_weights(client.model.get_weights(),client.model.get_weights())
