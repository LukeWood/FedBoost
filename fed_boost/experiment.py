import tensorflow.keras.datasets.cifar10 as cf
from fed_boost.data_extractor import alphas
from fed_boost.parameters import RANDOM, AVERAGE, AVERAGE_OUTPUT, server_epochs, v


class Experiment:
    def __init__(self, type, weak_learners_dir, results_dir):
        (_, _), (self.test_images, self.test_labels) = cf.load_data()
        self.alphas = alphas
        self.type = type

        self.weak_learners_dir = weak_learners_dir
        self.results_dir = results_dir

        if type == RANDOM:
            from fed_boost.random_server import RandomServer

            self.serverClass = RandomServer
        elif type == AVERAGE:
            from fed_boost.average_server import AverageServer

            self.serverClass = AverageServer
        elif type == AVERAGE_OUTPUT:
            from fed_boost.average_output_server import AverageOutputServer

            self.serverClass = AverageOutputServer
        else:
            from fed_boost.gd_boosted_Server import GDBoostServer

            self.serverClass = GDBoostServer

    def run(self, alpha):
        print(f"running experiment for {self.type} server")
        acc_all = {}
        # self.server.load_server(servers_dir)
        print(f"Creating {self.type} server with alpha {alpha}")
        self.server = self.serverClass(alpha, self.weak_learners_dir)
        print(f"Running experiment for alpha = {alpha}")
        self.server.train(server_epochs, v)
        acc = self.server.get_accuracy()
        print(f"Accuracy for {self.type} server for alpha = {alpha} is {acc}")
        print(f"Saving {self.type} server with alpha {alpha}")
        self.server.save_server(self.results_dir)
        return acc
