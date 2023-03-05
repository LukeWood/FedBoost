import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from fed_boost.server import Server
from fed_boost.parameters import (
    num_filters,
    filter_size,
    pool_size,
    input_image_shape,
    output_class_size,
)


class AverageServer(Server):
    def __init__(self, alpha, path):
        print(f"Creating Average Server")
        print(f"Initializing weak learners")
        super().__init__(alpha, path)
        print(f"Weak learners are loaded")
        len_weights = len(self.weak_learners)
        avg_weight = None
        for learner in self.weak_learners:
            avg_weight = learner.add_weights(
                learner.divide_weights(len_weights), avg_weight
            )
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

        model.set_weights(avg_weight)
        self.model = model
        print(f"server model ready")

    def save_server(self, path):
        self.model.save(f"{path}/average_server_al{self.data_alpha}.h5")

    def load_server(self, path):
        self.model.load_weights(f"{path}/average_server_al{self.data_alpha}.h5")

    def predict(self, x):
        if x.shape == input_image_shape:
            return np.argmax(self.model.predict(np.array([x])), axis=1)[0]
        else:
            return np.argmax(self.model.predict(x), axis=1)
