import random
from server import Server


class RandomServer(Server):
    def __init__(self, alpha, path=None):
        print(f"Creating Random Server")
        print(f"Initializing weak learners")
        super().__init__(alpha, path)
        print(f"Weak learners are loaded")
        print(f"server model ready")

    def save_server(self, path):
        for i in range(len(self.weak_learners)):
            self.weak_learners[i].save_model(f"{path}/random_server")

    def load_server(self, path):
        for i in range(len(self.weak_learners)):
            self.weak_learners[i].load_model(f"{path}/random_server")

    def predict(self, x):
        random_learner_index = random.sample(range(len(self.weak_learners)), 1)
        return self.weak_learners[random_learner_index[0]].predict(x)
