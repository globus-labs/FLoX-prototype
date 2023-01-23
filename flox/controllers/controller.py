import time
from collections import deque

import numpy as np
from funcx import FuncXClient

from flox.logic import FloxControllerLogic


class MainController(FloxControllerLogic):
    def __init__(
        self,
        endpoint_ids=None,
        num_samples=None,
        epochs=None,
        rounds=None,
        client_logic=None,
        global_model=None,
        model_trainer=None,
        path_dir=None,
        x_test=None,
        y_test=None,
        data_source=None,
        dataset_name=None,
        preprocess=None,
        x_train_filename=None,
        y_train_filename=None,
        input_shape=None,
        timeout=None,
    ):
        self.endpoint_ids = endpoint_ids
        self.num_samples = num_samples
        self.epochs = epochs
        self.rounds = rounds
        self.client_logic = client_logic
        self.global_model = global_model
        self.model_trainer = model_trainer
        self.path_dir = path_dir
        self.x_test = x_test
        self.y_test = y_test
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.preprocess = preprocess
        self.x_train_filename = x_train_filename
        self.y_train_filename = y_train_filename
        self.input_shape = input_shape
        self.timeout = timeout

    def on_model_init(self):
        # if num_samples or epochs is an int, convert to list so the same number can be applied to all endpoints
        if type(self.num_samples) == int:
            self.num_samples = [self.num_samples] * len(self.endpoint_ids)

        if type(self.epochs) == int:
            self.epochs = [self.epochs] * len(self.endpoint_ids)

        if type(self.path_dir) == str:
            self.path_dir = [self.path_dir] * len(self.endpoint_ids)

        if not self.timeout:
            self.timeout = 60

        self.funcx_client = FuncXClient(http_timeout=60)

    def create_config():
        pass

    def on_model_broadcast(self):
        # define list storage for results
        tasks = deque()

        # registr the function
        function_id = self.funcx_client.register_function(self.client_logic.run_round)

        # submit the corresponding parameters to each endpoint for a round of FL
        for ep, num_s, num_epoch, path_d in zip(
            self.endpoint_ids, self.num_samples, self.epochs, self.path_dir
        ):
            try:
                ep_status = self.funcx_client.get_endpoint_status(ep)["status"]
            except Exception as exp:
                print(
                    f"Could not check the status of the endpoint {ep}, the error is: {exp}"
                )
                ep_status = "error"

            if ep_status != "online":
                print(f"Endpoint {ep} is not online, it's {ep_status}!")
            else:
                config = self.create_config(num_s, num_epoch, path_d)

                task = self.funcx_client.run(
                    self.client_logic,
                    config,
                    self.model_trainer,
                    endpoint_id=ep,
                    function_id=function_id,
                )
                tasks.append(task)

            self.task_start_time = (
                time.time()
            )  # how would this work for multiple endpoints?

            if len(tasks) == 0:
                raise ValueError(
                    f"The tasks queue is empty, no tasks were submitted for training!"
                )

        return tasks

    def on_model_receive(self, tasks):
        # extract model updates from each endpoints once they are available
        model_weights = []
        samples_count = []

        while tasks and (time.time() - self.task_start_time) < self.timeout:
            t = tasks.popleft()
            if self.funcx_client.get_task(t)["status"] == "success":
                res = self.funcx_client.get_result(t)
                model_weights.append(res["model_weights"])
                samples_count.append(res["samples_count"])
            else:
                tasks.append(t)

        samples_count = np.array(samples_count)
        total = sum(samples_count)
        fractions = samples_count / total
        return {
            "model_weights": model_weights,
            "samples_count": samples_count,
            "bias_weights": fractions,
        }

    def on_model_aggregate(self, results):
        average_weights = np.average(
            results["model_weights"], weights=results["bias_weights"], axis=0
        )

        return average_weights

    def on_model_update(self, updated_weights) -> None:
        self.model_trainer.set_weights(self.global_model, updated_weights)

    def on_model_evaluate(self):
        if self.x_test is not None and self.y_test is not None:
            results = self.model_trainer.evaluate(
                self.global_model, self.x_test, self.y_test
            )
            print(results)
            return results
        else:
            print("Skipping evaluation, no x_test and/or y_test provided")
            return False

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
            self.on_model_evaluate()
