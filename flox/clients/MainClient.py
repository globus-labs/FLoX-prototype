import os
import platform
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from flox.common import NDArrays
from flox.logic import FloxClientLogic


class MainClient(FloxClientLogic):
    def on_model_receive(self):
        pass

    def retrieve_local_data(self, config: dict) -> tuple:
        """
        Retrieves x_train and y_train from given paths of .npy files
        Overwrite this function if you need to retrieve your locally stored data differently

        Parameters
        ----------
        config: dict
            dictionary with values indicating the path directory and file names for x_ and y_train data

        Returns
        -------
        raw_training_data: Tuple(NDArrays, NDArrays)
            x_train and y_train returned as a tuple

        """
        x_train_path_file = os.sep.join(
            [config["path_dir"], config["x_train_filename"]]
        )
        y_train_path_file = os.sep.join(
            [config["path_dir"], config["y_train_filename"]]
        )

        # load the files
        with open(x_train_path_file, "rb") as f:
            x_train = np.load(f)

        with open(y_train_path_file, "rb") as f:
            y_train = np.load(f)

        return (x_train, y_train)

    def retrieve_framework_data(self):
        raise NotImplementedError("Method not implemented")

    def on_data_retrieve(self, config):
        if config["data_source"] == "local":
            raw_training_data = self.retrieve_local_data(config)

        elif config["data_source"] == "framework":
            raw_training_data = self.retrieve_framework_data(config)

        else:
            # possibly allow for custom data_source and processing function?
            # otherwise throw an error that the data_source can be one of the two options
            raise KeyError("Please choose one of data sources: ['local', 'framework']")

        return raw_training_data

    def on_data_process(self, data, config):
        return data

    def on_model_send(
        self,
        fit_results,
        training_data=None,
        config=None,
        task_runtime=None,
        task_start_timestamp=None,
        task_finish_timestamp=None,
        data_processing_runtime=None,
        training_runtime=None,
        endpoint_physical_cores=None,
        endpoint_logical_cores=None,
        endpoint_physical_memory=None,
        platform_name=None,
    ):

        return {
            "model_weights": fit_results,
            "samples_count": self.get_number_of_samples(training_data, config),
            "task_runtime": task_runtime,
            "task_start_timestamp": task_start_timestamp,
            "task_finish_timestamp": task_finish_timestamp,
            "data_processing_runtime": data_processing_runtime,
            "training_runtime": training_runtime,
            "endpoint_physical_cores": endpoint_physical_cores,
            "endpoint_logical_cores": endpoint_logical_cores,
            "endpoint_physical_memory": endpoint_physical_memory,
            "endpoint_platform_name": platform_name,
        }

    def get_number_of_samples(self, training_data, config):
        return config.get("num_samples", None)

    def run_round(self, config, model_trainer):
        import platform
        from datetime import datetime
        from timeit import default_timer as timer

        import psutil

        task_start_time = timer()
        task_start_timestamp = datetime.utcnow()

        data_processing_start = timer()
        raw_training_data = self.on_data_retrieve(config)
        processed_training_data = self.on_data_process(raw_training_data, config)
        data_processing_runtime = timer() - data_processing_start

        training_start = timer()
        fit_results = self.on_model_fit(model_trainer, config, processed_training_data)
        training_runtime = timer() - training_start

        task_runtime = timer() - task_start_time
        task_finish_timestamp = datetime.utcnow()

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        physical_memory = psutil.virtual_memory().total
        platform_name = platform.node()

        task_results = self.on_model_send(
            fit_results,
            training_data=processed_training_data,
            config=config,
            task_runtime=task_runtime,
            task_start_timestamp=task_start_timestamp,
            task_finish_timestamp=task_finish_timestamp,
            data_processing_runtime=data_processing_runtime,
            training_runtime=training_runtime,
            endpoint_physical_cores=physical_cores,
            endpoint_logical_cores=logical_cores,
            endpoint_physical_memory=physical_memory,
            platform_name=platform_name,
        )

        return task_results
