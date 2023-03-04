import h5py
import numpy as np
from server import Server
from scipy.optimize import line_search
from parameters import client_size, output_class_size
from tensorflow.keras.utils import to_categorical


class GDBoostServer(Server):
    def loss(self, x, z):
        return (to_categorical(z, output_class_size) - self.w(x, z)) ** 2

    def gradient(self, x, z):
        return 2 * (to_categorical(z, output_class_size) - self.w(x, z))

    def __init__(self, alpha, path=None):
        print(f"Creating GDBoost Server")
        print(f"Initializing weak learners")
        super().__init__(alpha, path)
        self.af = np.full(client_size, 0.0)
        print(f"Weak learners are loaded")
        print(f"server model ready")

    def f(self, x):
        self.alpha_producer = np.linalg.pinv(
            np.array(
                list(map(lambda y: y.predict_without_softmax(x), self.weak_learners))
            ).T
        )
        return np.sum(
            np.array(
                list(
                    map(
                        lambda y: y[1].predict_without_softmax(x) * self.af[y[0]],
                        enumerate(self.weak_learners),
                    )
                )
            ),
            axis=0,
        )

    def w(self, x, z):
        f_x = self.f(x)
        result = np.exp(-1 * np.abs(f_x - f_x[z]) / 2)
        result[z] = np.sum(result)
        return result

    def alpha(self, x, z):
        w = self.w(x, z)
        f = self.f(x)
        return np.min(f / w)

    def one_iteration(self, v):
        af = np.zeros(output_class_size)
        mse = 0
        for i in range(len(self.train_labels)):
            x, z = self.train_images[i], self.train_labels[i]
            # Train a model with lse to get g
            # find alpha using linear search
            af += self.alpha(x, z) * self.w(x, z)
            mse += (np.linalg.norm(self.f(x) - self(x, z)) ** 2) / output_class_size
        self.mse = mse
        self.af = v * af

    def predict(self, x):
        return np.argmax(self.f(x), axis=0)

    def save_server(self, path):
        h5f = h5py.File(f"{path}/gdboost_server_extra_data_al{self.data_alpha}.h5", "w")
        h5f.create_dataset("alpha", data=self.af)
        h5f.create_dataset("mse_iter", data=self.mse_iter)
        h5f.create_dataset("acc_iter", data=self.acc_iter)
        h5f.close()
        for i in range(len(self.weak_learners)):
            self.weak_learners[i].save_model(f"{path}/gdboost_server")

    def load_server(self, path):
        h5f = h5py.File(f"{path}/gdboost_server_extra_data_al{self.data_alpha}.h5", "r")
        self.af = h5f["alpha"][:]
        self.af = h5f["mse_iter"][:]
        self.af = h5f["acc_iter"][:]
        h5f.close()
        for i in range(len(self.weak_learners)):
            self.weak_learners[i].load_model(f"{path}/gdboost_server")

    def train(self, Nb, v):
        self.mse_iter = np.zeros(Nb)
        self.acc_iter = np.zeros(Nb)
        print(f"Training Server")
        for i in range(Nb):
            self.one_iteration(v)
            self.mse_iter[i] = self.mse
            self.acc_iter[i] = np.sum(
                self.test_labels == self.predict(self.test_images)
            ) / len(self.test_labels)
            print("Training done")
            print(f"Loss after {i} iteration is {self.mse}")
        print(f"Training Server Done")
