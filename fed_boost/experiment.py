import tensorflow.keras.datasets.cifar10 as cf
from data_extractor import alphas
from parameters import RANDOM, AVERAGE, server_epochs, v, servers_dir, results_dir


class Experiment:
    def __init__(self, type):
        (_, _), (self.test_images, self.test_labels) = cf.load_data()
        self.alphas = alphas
        self.type = type
        if type == RANDOM:
            from random_server import RandomServer

            self.serverClass = RandomServer
        elif type == AVERAGE:
            from average_server import AverageServer

            self.serverClass = AverageServer
        else:
            from gd_boosted_Server import GDBoostServer

            self.serverClass = GDBoostServer

    def run(self, alpha):
        print(f"running experiment for {self.type} server")
        acc_all = {}
        # self.server.load_server(servers_dir)
        print(f"Creating {self.type} server with alpha {alpha}")
        self.server = self.serverClass(alpha)
        print(f"Running experiment for alpha = {alpha}")
        self.server.train(server_epochs, v)
        acc = self.server.get_accuracy()
        print(f"Accuracy for {self.type} server for alpha = {alpha} is {acc}")
        print(f"Saving {self.type} server with alpha {alpha}")
        self.server.save_server(servers_dir)
        return acc
