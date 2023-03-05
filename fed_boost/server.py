from fed_boost.client import Client
from fed_boost.parameters import client_size, client_epochs
import tensorflow.keras.datasets.cifar10 as cf


class Server:
    def __init__(self, alpha, path):
        self.data_alpha = alpha
        self.weak_learners = []
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = cf.load_data()
        # Normalize the images.
        self.train_images = (self.train_images / 255) - 0.5
        self.test_images = (self.test_images / 255) - 0.5
        for client_id in range(client_size):
            print(f"loading client model {client_id} for alpha = {alpha}")
            client = Client(client_id, client_epochs, alpha)
            client.load_model(path)
            self.weak_learners.append(client)
            print(f"client model loaded")

    def train(self, Nb, v):
        print(f"Training done")

    def predict(self, x):
        return 0

    def get_accuracy(self):
        acc = 0
        for x, z in zip(self.test_images, self.test_labels):
            z_pred = self.predict(x)
            acc += z == z_pred
        return acc / len(self.test_labels)
