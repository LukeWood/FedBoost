from experiment import Experiment
from parameters import RANDOM, AVERAGE, GDBOOST
from data_extractor import alphas

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="federated_learner", description="Federated Learning Experimenter"
    )

    parser.add_argument(
        "type",
        choices=[RANDOM, AVERAGE, GDBOOST],
        help=f"Choice of the type of server among 3 implementations\n1) {RANDOM}\n2) {AVERAGE}\n3) {GDBOOST}",
    )
    parser.add_argument(
        "alpha",
        choices=alphas,
        help=f"Choice of the alphas of datasplit among 8 possible values\n {alphas}\n",
    )

    args = parser.parse_args()
    server = Experiment(args.type)
    server.run(args.alpha)


# random_server = Experiment(RANDOM)

# average_server = Experiment(AVERAGE)

# gdboost_server = Experiment(GDBOOST)

# # random_server.run(alphas[1])
# # average_server.run(alphas[0])
# gdboost_server.run(alphas[0])

if __name__ == "__main__":
    main()
