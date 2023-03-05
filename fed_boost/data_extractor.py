import os.path as path
import csv
import numpy as np
import tensorflow.keras.datasets.cifar10 as cf
from fed_boost.parameters import client_size as cl_size

alphas = ["0.00", "0.05", "0.10", "0.20", "0.50", "1.00", "10.00", "100.00"]


def cifar_parser(line):
    user_id, image_id, class_id = line
    return user_id, image_id, class_id


def get_split_data(alpha,client_size=cl_size):
    (x_train, y_train), (x_test, y_test) = cf.load_data()
    dir_path = path.dirname(path.realpath(__file__))
    with open(
        path.join(dir_path, "cifar10_v1.1", f"federated_train_alpha_{alpha}.csv")
    ) as f:
        reader = csv.reader(f)
        next(reader)  # skip header.
        data = {}
        for line in reader:
            user_id, image_id, class_id = cifar_parser(line)
            client_id = int(user_id)%client_size
            if client_id not in data:
                data[client_id] = {"image_ids": [], "class_ids": []}
            data[client_id]["image_ids"].append(int(image_id))
            data[client_id]["class_ids"].append(int(class_id))
        for user_id in data:
            data[int(user_id)]["x_train"] = np.take(
                x_train, data[user_id]["image_ids"], 0
            )
            data[int(user_id)]["y_train"] = np.take(
                y_train, data[user_id]["image_ids"], 0
            )
        return data


# data_split = get_split_data(alphas[0],10)
# print(data_split.keys())
# print(data_split[0]['x_train'])
# print(data_split[0]['class_ids'])
