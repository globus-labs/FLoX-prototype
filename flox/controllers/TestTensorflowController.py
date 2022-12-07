import numpy as np
from funcx import FuncXExecutor

from flox.logic import FloxControllerLogic


class TestTensorflowController(FloxControllerLogic):
    def __init__(
        self,
        endpoint_ids=None,
        num_samples=None,
        epochs=None,
        rounds=None,
        ClientLogic=None,
        global_model=None,
        ModelTrainer=None,
        path_dir=None,
        x_test=None,
        y_test=None,
        data_source=None,
        dataset_name=None,
        preprocess=None,
    ):
        self.endpoint_ids = endpoint_ids
        self.num_samples = num_samples
        self.epochs = epochs
        self.rounds = rounds
        self.ClientLogic = ClientLogic
        self.global_model = global_model
        self.ModelTrainer = ModelTrainer
        self.path_dir = path_dir
        self.x_test = x_test
        self.y_test = y_test
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.preprocess = preprocess

    def on_model_init(self):
        # if num_samples or epochs is an int, convert to list so the same number can be applied to all endpoints
        if type(self.num_samples) == int:
            self.num_samples = [self.num_samples] * len(self.endpoint_ids)

        if type(self.epochs) == int:
            self.epochs = [self.epochs] * len(self.endpoint_ids)

    def on_model_broadcast(self):
        """DocString"""
        # get the model's architecture
        model_architecture = self.ModelTrainer.get_architecture(self.global_model)
        model_weights = self.ModelTrainer.get_weights(self.global_model)

        # define list storage for results
        tasks = []

        # submit the corresponding parameters to each endpoint for a round of FL
        for ep, num_s, num_epoch, path_d in zip(
            self.endpoint_ids, self.num_samples, self.epochs, self.path_dir
        ):
            config = {
                "num_samples": num_s,
                "epochs": num_epoch,
                "path_dir": path_d,
                "data_source": self.data_source,
                "dataset_name": self.dataset_name,
                "preprocess": self.preprocess,
                "architecture": model_architecture,
                "weights": model_weights,
            }
            with FuncXExecutor(endpoint_id=ep) as fx:
                task = fx.submit(
                    self.ClientLogic.run_round,
                    self.ClientLogic,
                    config,
                    self.ModelTrainer,
                )
                tasks.append(task)

        return tasks

    def on_model_receive(self, tasks):
        """DocString"""
        # extract model updates from each endpoints once they are available
        model_weights = [t.result()["model_weights"] for t in tasks]
        samples_count = np.array([t.result()["samples_count"] for t in tasks])

        total = sum(samples_count)
        fractions = samples_count / total
        return {
            "model_weights": model_weights,
            "samples_count": samples_count,
            "bias_weights": fractions,
        }

    def on_model_aggregate(self, results):
        """DocString"""
        average_weights = np.average(
            results["model_weights"], weights=results["bias_weights"], axis=0
        )

        return average_weights

    def on_model_update(self, updated_weights) -> None:
        """DocString"""
        self.ModelTrainer.set_weights(self.global_model, updated_weights)

    def on_model_evaluate(self, x_test, y_test):
        results = self.ModelTrainer.evaluate(self.global_model, x_test, y_test)
        print(results)
        return results

    def run_federated_learning(self):
        self.on_model_init()
        # start running FL loops
        for i in range(self.rounds):
            # broadcast the model
            tasks = self.on_model_broadcast()

            # process & decrypt the results
            results = self.on_model_receive(tasks)
            # aggregate the weights
            updated_weights = self.on_model_aggregate(results)

            # update the model's weights
            self.on_model_update(updated_weights)

            # evaluate the model
            print(f"Round {i} evaluation results: ")
            self.on_model_evaluate(self.x_test, self.y_test)
