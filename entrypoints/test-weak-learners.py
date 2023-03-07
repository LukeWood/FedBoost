# python train-weak-learners.py --model_path='weak_learners/'

import argparse
from fed_boost.experiment import Experiment
from fed_boost.parameters import RANDOM, AVERAGE, GDBOOST
from fed_boost.data_extractor import alphas
from fed_boost.client import Client
from fed_boost.parameters import client_size, client_epochs


def main():
    parser = argparse.ArgumentParser(
        prog="federated_learner", description="Federated Learning Experimenter"
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help=f"Which directory to load weak learners to.",
    )

    args = parser.parse_args()

    for alpha in alphas:
        for i in range(client_size):
            client = Client(i, client_epochs, alpha)
            client.load_model(args.model_path)
            acc = client.get_accuracy()
            print(f"Accuracy for Client {i} for alpha = {alpha} is {acc}")



if __name__ == "__main__":
    main()
