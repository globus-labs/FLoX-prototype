import concurrent.futures
import csv
import logging
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from funcx import FuncXClient, FuncXExecutor

from flox.common.flox_dataclasses import EndpointData, TaskData
from flox.common.logging_config import setup_logging
from flox.common.typing import NDArrays
from flox.logic import BaseModelTrainer, FloxClientLogic, FloxControllerLogic
from flox.utils import create_csv, write_to_csv

setup_logging(debug=True)
logger = logging.getLogger(__name__)


class MainController(FloxControllerLogic):
    """MainController implements the main controller functionality while allowing for
    extensibility by inhereting functionality from it

    endpoint_ids: List[str]
        a list with endpoint_ids to include in the FL process.
        The ids can be retrieved by running 'funcx-endpoint list' on participating devices.
        If using a non-funcx executor, pass *unique* arbitrary string values, such as ["fake_ep1", "fake_ep2"]

    client_logic: FloxClientLogic instance
        a class that implements abstract class methods from FloxClientLogic

    model_trainer: BaseModelTrainer instance
        a class that implements abstract class methods from BaseModelTrainer

    # TODO: should I move this parameter under those client_logic classes that actually use them?
    global_model: ML model object
       ML model that will be deployed for training on the endpoints.
       However, some ML frameworks might instead benefit from storing the model as
       an attribute of model_trainer, so global_model defaults to None and is not required.

    executor: concurrent.futures.Executor instance
        a class similar to concurrent.futures.Executor class that implements methods
        .submit() and returns a future-like class. MainController provides two executors
        to choose from (FuncXExecutor, ThreadPoolExecutor) although users may provide custom executors

    executor_type: str
        you can provide "funcx" or "local" if you wish to use one of the offered executors
        (FuncXExecutor, ThreadPoolExecutor). Otherwise, it will default to "local"

    num_samples: int or list
        indicates how many samples to get for training on endpoints
        if int, applies the same num_samples to all endpoints.
        if list, it will use the corresponding number of samples for each device
        the list should have the same number of entries as the number of endpoints

    epochs: int or list
        indicates how many epochs to use for training on endpoints
        if int, applies the same number of epochs to all endpoints.
        if list, it will use the corresponding number of epochs for each device
        the list should have the same number of entries as the number of endpoints

    rounds: int
        defines how many FL rounds to run. Each round consists of deploying the model, training,
        aggregating the updates, and reassigning new weights to the model.

    timeout: int
        defines the timeout after which the MainController will stop trying to retrieve tasks,
        assuming that the device has dropped out from training midway.

    data_source: str
        type of the data source supported by the provided client_logic instance

    dataset_name: str
        the dataset name supported by the provided client_logic instance

    # TODO: should I move this parameter under those client_logic classes that actually use them?
    preprocess: boolean
        used to indicate whether to preprocess data on the client side, if supported by client_logic

    path_dir: str
        path to the folder with x_train_filename and y_train_filename; needed when data_sourse="local"

    x_train_filename: str
        filename for x_train; needed when data_sourse="local"

    y_train_filename: str
        file name for y_train; needed when data_sourse="local"

    # TODO: should I move this parameter under those client_logic classes that actually use them?
    input_shape: tupple
        input shape for the provided model

    x_test: list/numpy array/tensors
        x_test data for testing

    y_test: list
        y_test labels for x_test

    running_average: bool
        if True, will use `tasks_to_running_average` to aggregate the model weights
        as a simple average on the fly as they are returned from the Executor.
        This speeds up the total processing time when you have many endpoints and aggregating
        all of them in a single operation takes too long

    """

    AVAILABLE_EXECUTORS = {"local": ThreadPoolExecutor, "funcx": FuncXExecutor}
    # TODO: verify the header columns
    CSV_HEADER = [
        "experiment_id",
        "experiment_name",
        "experiment_description",
        "experiment_executor",
        "dataset_name",
        "data_source",
        "round_number",
        "n_clients_provided",
        "n_clients_participated",
        "round_aggregated_accuracy",
        "round_aggregated_loss",
        "round_aggregation_runtime",
        "total_round_runtime",
        "round_start_time",
        "round_end_time",
        "endpoint_uuid",
        "endpoint_custom_name",
        "endpoint_latest_status",
        "endpoint_ram",
        "endpoint_physical_cores",
        "endpoint_logical_cores",
        "endpoint_platform_name",
        "number_of_tasks",
        "desired_n_samples",
        "epochs",
        "batch_size",
        "task_local_uuid",
        "task_funcx_uuid",
        "file_size",
        "task_actual_n_samples",
        "task_model_accuracy",
        "task_model_loss",
        "task_runtime",
        "task_training_runtime",
        "task_data_processing_runtime",
        "task_broadcasted_to_retrieved_runtime",
        "task_broadcast_start_time",
        "task_broadcast_finish_time",
        "task_start_time",
        "task_finish_time",
        "task_results_received_time",
    ]

    def __init__(
        self,
        endpoint_ids: List[str],
        client_logic: FloxClientLogic,
        model_trainer: BaseModelTrainer,
        global_model: NDArrays = None,
        executor: concurrent.futures.Executor = None,
        executor_type: str = "local",
        num_samples: Union[int, List[int]] = 100,
        epochs: Union[int, List[int]] = 5,
        batch_size: Union[int, List[int]] = None,
        rounds: int = 3,
        path_dir: Union[str, List[int]] = ".",
        x_test=None,
        y_test=None,
        data_source: str = None,
        dataset_name=None,
        preprocess: bool = True,
        x_train_filename: str = None,
        y_train_filename: str = None,
        input_shape: Tuple[int] = None,
        timeout: int = float("inf"),
        running_average: bool = False,
        tasks_per_endpoint: Union[int, List[int]] = 1,
        csv_filename: str = None,
    ):
        self.endpoint_ids = endpoint_ids
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.rounds = rounds
        self.client_logic = client_logic
        self.global_model = global_model
        self.model_trainer = model_trainer
        self.executor = executor
        self.executor_type = executor_type
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
        self.running_average = running_average
        self.tasks_per_endpoint = tasks_per_endpoint
        self.csv_filename = csv_filename
        self.funcx_client = FuncXClient(http_timeout=60)
        self.endpoints = []
        self.tasks = {}

    def on_model_init(self) -> None:
        """Does initial Controller setup before running the main Federated Learning loop"""
        # if num_samples or epochs is an int, convert to list so the same number can be applied to all endpoints
        if type(self.num_samples) == int:
            self.num_samples = [self.num_samples] * len(self.endpoint_ids)

        if type(self.epochs) == int:
            self.epochs = [self.epochs] * len(self.endpoint_ids)

        if type(self.tasks_per_endpoint) == int:
            self.tasks_per_endpoint = [self.tasks_per_endpoint] * len(self.endpoint_ids)

        if type(self.path_dir) == str:
            self.path_dir = [self.path_dir] * len(self.endpoint_ids)

        if type(self.batch_size) != list:
            self.batch_size = [self.batch_size] * len(self.endpoint_ids)

        if not self.executor:
            logger.debug(
                f"No executor was provided, trying to retrieve the provided executor type {self.executor_type} from the list of available executors: {self.AVAILABLE_EXECUTORS}"
            )
            try:
                self.executor = self.AVAILABLE_EXECUTORS[self.executor_type]
                logger.debug(f"The selected executor is {self.executor}")
            except KeyError as exp:
                logger.debug(
                    f"Could not find {self.executor_type} in the list of available executors: {self.AVAILABLE_EXECUTORS}. Please provide your own executor or select from {self.AVAILABLE_EXECUTORS.keys()}"
                )
                self.executor = self.AVAILABLE_EXECUTORS["local"]
                logger.debug(
                    f"Defaulting to {self.AVAILABLE_EXECUTORS['local']} executor"
                )
        if self.csv_filename:
            create_csv(filename=self.csv_filename, header=self.CSV_HEADER)

        for ep, num_s, num_epoch, path_d, n_tasks, batch_s in zip(
            self.endpoint_ids,
            self.num_samples,
            self.epochs,
            self.path_dir,
            self.tasks_per_endpoint,
            self.batch_size,
        ):
            self.endpoints.append(
                EndpointData(
                    id=ep,
                    tasks_per_endpoint=n_tasks,
                    desired_n_samples=num_s,
                    epochs=num_epoch,
                    batch_size=batch_s,
                    path_directory=path_d,
                )
            )

    def create_config(self, *args, **kwargs) -> Dict:
        """Creates a config dictionary that will be broadcasted in self.on_model_broadcast()
        to the endpoints. The methods requires the user to extend it since different ML frameworks
        require a different set of arguments.

        Returns
        -------
        config: Dict
            a dictionary of parameters required for use by client_logic.run_round()

        """
        raise NotImplementedError("Method not implemented")

    def on_model_broadcast(self) -> deque:
        """Sends the model and config to endpoints for FL training.

        Returns
        -------
        tasks: deque[future]
            return a list of futures. The futures should support methods .done() and .result()

        """
        # define list storage for results
        # tasks = deque()

        logger.debug(f"Launching the {self.executor} executor")
        with self.executor() as executor:
            # submit the corresponding parameters to each endpoint for a round of FL
            assert (
                len(self.endpoint_ids)
                == len(self.num_samples)
                == len(self.epochs)
                == len(self.path_dir)
            )
            for ep in self.endpoints:
                logger.info(f"Starting to broadcast a task to endpoint {ep.id}")
                try:
                    ep.latest_status = self.funcx_client.get_endpoint_status(ep.id)[
                        "status"
                    ]
                except Exception as exp:
                    logger.warning(
                        f"Could not check the status of the endpoint {ep.id}, the error is: {exp}"
                    )
                    ep.latest_status = "error"

                if ep.latest_status != "online" and self.executor == FuncXExecutor:
                    logger.warning(
                        f"Endpoint {ep.id} is not online, it's {ep.latest_status}!"
                    )
                else:
                    config = self.create_config(
                        num_s=ep.desired_n_samples,
                        num_epoch=ep.epochs,
                        path_d=ep.path_directory,
                        batch_size=ep.batch_size,
                    )
                    executor.endpoint_id = ep.id

                    for i in self.tasks_per_endpoint:
                        task_data = TaskData()
                        task_data.broadcast_start_timestamp = datetime.utcnow()
                        if self.executor_type == "local":
                            future = executor.submit(
                                self.client_logic.run_round, config, self.model_trainer
                            )

                        elif self.executor_type == "funcx":
                            future = executor.submit(
                                self.client_logic.run_round,
                                self.client_logic,  # funcxExecutor requires the class submitted as well, while the ThreadPoolExecutor does not
                                config,
                                self.model_trainer,
                            )
                        else:
                            raise ValueError(
                                f"{self.executor_type} is invalid executor type, choose one of [local, funcx]"
                            )

                        task_data.future = future
                        task_data.broadcast_finish_timestamp = datetime.utcnow()
                        task_data.local_id = uuid.uuid1()
                        self.tasks[task_data.local_id] = task_data
                        ep.task_ids.append(task_data.local_id)
                        logger.info(f"Deployed task {i} to endpoint {ep.id}")

            self.task_start_time = (
                timer()
            )  # how would this work for multiple endpoints?

            if len(self.tasks) == 0:
                logger.error(
                    f"The tasks queue is empty, here are the endpoints' statuses: {[ep.latest_status for ep in self.endpoints]}"
                )
                raise ValueError(
                    f"The tasks queue is empty, no tasks were submitted for training!"
                )

        return self.tasks

    def on_model_receive(self, tasks: dict) -> Dict:
        """Processes returned tasks from on_model_broadcast.

        Parameters
        ----------
        tasks: List[futures]
            A list of tasks/futures with results of the FL training returned from the endpoints.
            If using FuncXExecutor/ThreadPoolExecutor, this would be a list of futures
            funcX/ThreadPoolExecutor returns after you submit functions to endpoints.

        Returns
        -------
        results: Dict
            FL results extracted from tasks, formatted as a dictionary. For example:
            results = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
            }
        """
        model_weights = []
        samples_count = []
        n_clients_participated = 0

        tasks_queue = deque(list(tasks.keys()))
        logger.info("Starting to retrieve results from endpoints")
        while tasks_queue and (timer() - self.task_start_time) < self.timeout:
            task_id = tasks_queue.popleft()
            task_data = tasks[task_id]
            logger.warning("Popped the task")
            logger.warning(f"This task's future is {task_data.future}")
            if task_data.future.done():
                logger.warning("Task is done, starting to retrieve it")
                task_data.future_completed_timestamp = datetime.utcnow()

                res = task_data.future.result()
                logger.warning(f"Retrieved task {task_data.future}")

                model_weights.append(res["model_weights"])
                task_samples = res.get("samples_count", None)
                samples_count.append(task_samples)
                n_clients_participated += 1

                try:
                    task_data.funcx_uuid = task_data.future.task_id
                except:
                    task_data.funcx_uuid = np.nan

                task_data.n_samples = task_samples
                task_data.task_runtime = res["task_runtime"]
                task_data.task_start_timestamp = res["task_start_timestamp"]
                task_data.task_finish_timestamp = res["task_finish_timestamp"]
                task_data.data_processing_runtime = res["data_processing_runtime"]
                task_data.training_runtime = res["training_runtime"]
                task_data.physical_memory = res["endpoint_physical_memory"]
                task_data.physical_cores = res["endpoint_physical_cores"]
                task_data.logical_cores = res["endpoint_logical_cores"]
                task_data.endpoint_platform_name = res["endpoint_platform_name"]
                task_data.actual_n_samples = res["samples_count"]
                task_data.broadcasted_to_retrieved_runtime = (
                    task_data.future_completed_timestamp
                    - task_data.broadcast_finish_timestamp
                ).total_seconds()
            else:
                tasks_queue.append(task_data)
                logger.info(f"Retrieved results from endpoints {task_data.future}")

        samples_count = np.array(samples_count)
        total = sum(samples_count)
        fractions = samples_count / total
        logger.info("Finished retrieving all results from the endpoints")
        return {
            "model_weights": model_weights,
            "samples_count": samples_count,
            "bias_weights": fractions,
            "n_clients_participated": n_clients_participated,
        }

    def on_model_aggregate(self, results: Dict) -> NDArrays:
        """Aggregates weights.

        Parameters
        ----------
        results: Dict
            FL results returned by on_model_receive, formatted as a dictionary. For example:
            results = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
            }

        Returns
        -------
        average_weights: NDArrays
            ML model weights for a single model in the form of Numpy Arrays.

        """
        aggregaton_start_time = timer()
        logger.info(f"Aggregating {len(results['model_weights'])} weights")
        average_weights = np.average(
            results["model_weights"], weights=results["bias_weights"], axis=0
        )
        logger.info("Finished aggregating weights")
        aggregation_runtime = timer() - aggregaton_start_time

        return average_weights, aggregation_runtime

    def on_model_update(self, updated_weights: NDArrays) -> None:
        """Updates the model's weights with new weights.
        The method might need to be overriden depending on the Machine Learning framework as use.

        Parameters
        ----------
        updated_weights: NDArrays
            ML model weights for a single model in the form of Numpy Arrays.

        """
        self.model_trainer.set_weights(self.global_model, updated_weights)
        logger.info("Updated the global model's weights")

    def on_model_evaluate(self):
        """Evaluates the given model using provided test data.
        The method might need to be overriden depending on the Machine Learning framework as use.

        Returns
        -------
        EvaluateRes
            A class consisting of two attributes showing the loss and metrics of the evaluated
            model on the dataset:
                loss: float
                metrics: Dict[str, Scalar]
        """
        if self.x_test is not None and self.y_test is not None:
            logger.info("Starting evaluation")
            results = self.model_trainer.evaluate(
                self.global_model, self.x_test, self.y_test
            )
            logger.info(f"Evaluation results: {results}")
            return results
        else:
            logger.warning("Skipping evaluation, no x_test and/or y_test provided")
            return False

    def run_federated_learning(
        self,
        experiment_id: int = uuid.uuid1(),
        experiment_name: str = np.nan,
        experiment_description: str = np.nan,
    ):
        """The main Federated Learning loop that ties all the functions together.
        Runs <self.rounds> rounds of federated learning"""
        self.on_model_init()
        # start running FL loops
        for i in range(self.rounds):
            round_start_time = timer()
            round_start_timestamp = datetime.utcnow()

            # broadcast the model
            tasks = self.on_model_broadcast()

            if self.running_average:
                updated_weights = self.tasks_to_running_average(tasks)[
                    "running_average_weights"
                ]
            else:
                # process & decrypt the results
                results = self.on_model_receive(tasks)

                # aggregate the weights
                updated_weights, aggregation_runtime = self.on_model_aggregate(results)

            # update the model's weights
            self.on_model_update(updated_weights)

            total_round_runtime = timer() - round_start_time
            round_end_timestamp = datetime.utcnow()

            # evaluate the model
            logger.info(f"Round {i} evaluation results: ")
            evaluation_results = self.on_model_evaluate()

            # store results in .csv
            if self.csv_filename:
                self.record_experiment(
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                    experiment_description=experiment_description,
                    round_number=i,
                    retrieved_results=results,
                    tasks_data=self.tasks,
                    endpoints_data=self.endpoints,
                    evaluation_results=evaluation_results,
                    aggregation_runtime=aggregation_runtime,
                    total_round_runtime=total_round_runtime,
                    round_start_timestamp=round_start_timestamp,
                    round_end_timestamp=round_end_timestamp,
                )

    def record_experiment(
        self,
        experiment_id,
        experiment_name,
        experiment_description,
        round_number,
        retrieved_results,
        tasks_data,  # take most of Client-side values from here, and use the weights to evaluate the individual weights
        endpoints_data,
        evaluation_results,
        aggregation_runtime,
        total_round_runtime,
        round_start_timestamp,
        round_end_timestamp,
    ):
        # evaluate individual models

        # unpack lists & dics
        rows = []
        logger.info("Starting to create data rows")
        for ep in endpoints_data:
            for task_id in ep.task_ids:
                task = tasks_data[task_id]
                data_entry = {
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "experiment_description": experiment_description,
                    "experiment_executor": self.executor_type,
                    "dataset_name": self.dataset_name,
                    "data_source": self.data_source,
                    "round_number": round_number,
                    "n_clients_provided": len(self.endpoints),
                    "n_clients_participated": retrieved_results.get(
                        "n_clients_participated", None
                    ),
                    "round_aggregated_accuracy": evaluation_results["metrics"][
                        "accuracy"
                    ],
                    "round_aggregated_loss": evaluation_results["loss"],
                    "round_aggregation_runtime": aggregation_runtime,
                    "total_round_runtime": total_round_runtime,
                    "round_start_time": round_start_timestamp,
                    "round_end_time": round_end_timestamp,
                    "endpoint_uuid": ep.id,
                    "endpoint_custom_name": ep.endpoint_custom_name,
                    "endpoint_latest_status": ep.latest_status,
                    "endpoint_ram": task.physical_memory,
                    "endpoint_physical_cores": task.physical_cores,
                    "endpoint_logical_cores": task.logical_cores,
                    "endpoint_platform_name": task.endpoint_platform_name,
                    "number_of_tasks": ep.tasks_per_endpoint,
                    "desired_n_samples": ep.desired_n_samples,
                    "epochs": ep.epochs,
                    "batch_size": ep.batch_size,
                    "task_local_uuid": task.local_id,
                    "task_funcx_uuid": task.funcx_uuid,
                    "file_size": task.file_size,
                    "task_actual_n_samples": task.actual_n_samples,
                    "task_model_accuracy": task.model_accuracy,
                    "task_model_loss": task.model_loss,
                    "task_runtime": task.task_runtime,
                    "task_training_runtime": task.training_runtime,
                    "task_data_processing_runtime": task.data_processing_runtime,
                    "task_broadcasted_to_retrieved_runtime": task.broadcasted_to_retrieved_runtime,
                    "task_broadcast_start_time": task.broadcast_start_timestamp,
                    "task_broadcast_finish_time": task.broadcast_finish_timestamp,
                    "task_start_time": task.task_start_timestamp,
                    "task_finish_time": task.task_finish_timestamp,
                    "task_results_received_time": task.future_completed_timestamp,
                }
                rows.append(data_entry)
        # write to csv
        logger.info(f"Starting to write data rows to {self.csv_filename}")
        with open(self.csv_filename, "a", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            for row in rows:
                writer.writerow(row)

    def tasks_to_running_average(self, tasks: deque):
        """Serves as a modified combination of on_model_receive and on_model_aggregate.
        Aggregates the model weights
        as a simple average on the fly as they are returned from the Executor.
        This speeds up the total processing time when you have many endpoints and aggregating
        all of them in a single operation takes too long. Note that the aggregated model
        will be slightly different from what you would get if you used on_model_aggregate
        on all of the weights in a single operation. This is because of the reiterative
        floating point averaging.

        Parameters
        ----------
        tasks: List[futures]
            A list of tasks/futures with results of the FL training returned from the endpoints.
            If using FuncXExecutor/ThreadPoolExecutor, this would be a list of futures
            funcX/ThreadPoolExecutor returns after you submit functions to endpoints.

        Returns
        -------
        results: Dict
            FL results extracted from tasks, formatted as a dictionary. For example:
            results = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
                "running_average_weights": running_average
            }
        """
        model_weights = []
        samples_count = []

        n_aggregated = 0
        n_total = 0
        running_average = None

        logger.info("Starting to retrieve results from endpoints")
        while tasks and (timer() - self.task_start_time) < self.timeout:
            t = tasks.popleft()
            if t.done():
                res = t.result()
                new_weights = res["model_weights"]
                model_weights.append(new_weights)
                samples_count.append(res["samples_count"])
                # update running average
                if running_average is None:
                    logger.debug(
                        "the running average is NONE, instantiating it for the first time"
                    )
                    running_average = new_weights
                    n_aggregated += 1
                    n_total += 1
                else:
                    n_total += 1
                    running_weights = [
                        (n_aggregated / n_total),
                        (n_total - n_aggregated) / n_total,
                    ]
                    logger.info(
                        f"The weights are {running_weights} and sum up to {sum(running_weights)}"
                    )
                    running_average = np.average(
                        [running_average, new_weights], weights=running_weights, axis=0
                    )
                    n_aggregated += 1

            else:
                tasks.append(t)
                logger.info(f"Retrieved results from endpoints {t}")

        samples_count = np.array(samples_count)
        total = sum(samples_count)
        fractions = samples_count / total
        logger.info("Finished retrieving all results from the endpoints")
        return {
            "model_weights": model_weights,
            "samples_count": samples_count,
            "bias_weights": fractions,
            "running_average_weights": running_average,
        }
