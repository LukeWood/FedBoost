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
    client_size
)


class AverageOutputServer(Server):
    def __init__(self, alpha, path):
        print(f"Creating Average Server")
        print(f"Initializing weak learners")
        super().__init__(alpha, path)
        print(f"Weak learners are loaded")
        print(f"server model ready")

    def save_server(self, path):
        print('no need to save as this is same for random')

    def load_server(self, path):
        print('no need to load as this is same for random')

    def predict(self, x):
        # random_learner_index = random.sample(range(len(self.weak_learners)), 1)
        # return self.weak_learners[random_learner_index[0]].predict(x)
        # import pdb; pdb.set_trace()
        avg_score = np.zeros(output_class_size)
        for weak_learners in self.weak_learners:
            avg_score += weak_learners.model.predict(np.array([x]))[0]
        return np.argmax(avg_score)
            
