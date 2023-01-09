import concurrent.futures

import numpy as np

from flox.controllers.PyTorchController import PyTorchController


class PyTorchControllerLocal(PyTorchController):
    def on_model_broadcast(self):
        # define list storage for results
        tasks = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # submit the corresponding parameters to each endpoint for a round of FL
            for ep, num_s, num_epoch, path_d in zip(
                self.endpoint_ids, self.num_samples, self.epochs, self.path_dir
            ):
                config = {
                    "num_samples": num_s,
                    "epochs": num_epoch,
                    "path_dir": path_d,
                    "dataset_name": self.dataset_name,
                    "preprocess": self.preprocess,
                }
                print("Starting task submission...")

                task = executor.submit(
                    self.client_logic.run_round,
                    config,
                )
                tasks.append(task)
                print("SENT TASKS TO EXECUTOR")

        return tasks

    def on_model_receive(self, tasks):
        # extract model updates from each endpoints once they are available
        print("Starting to UNPACK tasks...")
        model_weights = [t.result()["model_weights"] for t in tasks]
        samples_count = np.array([t.result()["samples_count"] for t in tasks])

        total = sum(samples_count)
        fractions = samples_count / total
        print("UNPACKED tasks, returning weights...")
        return {
            "model_weights": model_weights,
            "samples_count": samples_count,
            "bias_weights": fractions,
        }

    def on_model_update(self, updated_weights) -> None:
        self.client_logic.set_weights(updated_weights)

    def on_model_evaluate(self, testloader):
        results = self.client_logic.evaluate(testloader)
        print(results)
        return results
