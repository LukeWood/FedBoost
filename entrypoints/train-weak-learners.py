# python train-weak-learners.py --model_path='models/'

from fed_boost.experiment import Experiment
from fed_boost.parameters import RANDOM, AVERAGE, GDBOOST
from fed_boost.data_extractor import alphas
from fed_boost.client import Client


def main():
    parser = argparse.ArgumentParser(
        prog="federated_learner", description="Federated Learning Experimenter"
    )

    parser.add_argument(
        "model_path",
        required=True,
        help=f"Which model to save weak learners to.",
    )

    args = parser.parse_args()

    for alpha in alphas:
        for i in range(client_size):
            client = Client(i,client_epochs,alpha)
            client.train_model()
            client.save_model(args.model_path)

if __name__ == "__main__":
    main()
