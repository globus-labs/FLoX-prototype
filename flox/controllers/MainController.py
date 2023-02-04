import concurrent.futures
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from funcx import FuncXClient, FuncXExecutor

from flox.common.logging_config import setup_logging
from flox.common.typing import NDArrays
from flox.logic import BaseModelTrainer, FloxClientLogic, FloxControllerLogic

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

    """

    AVAILABLE_EXECUTORS = {"local": ThreadPoolExecutor, "funcx": FuncXExecutor}

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
        rounds: int = 3,
        path_dir: Union[str, List[int]] = ["."],
        x_test=None,
        y_test=None,
        data_source: str = None,
        dataset_name=None,
        preprocess: bool = True,
        x_train_filename: str = None,
        y_train_filename: str = None,
        input_shape: Tuple[int] = None,
        timeout: int = 60,
    ):
        self.endpoint_ids = endpoint_ids
        self.num_samples = num_samples
        self.epochs = epochs
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

    def on_model_init(self) -> None:
        """Does initial Controller setup before running the main Federated Learning loop"""
        # if num_samples or epochs is an int, convert to list so the same number can be applied to all endpoints
        if type(self.num_samples) == int:
            self.num_samples = [self.num_samples] * len(self.endpoint_ids)

        if type(self.epochs) == int:
            self.epochs = [self.epochs] * len(self.endpoint_ids)

        if type(self.path_dir) == str:
            self.path_dir = [self.path_dir] * len(self.endpoint_ids)

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

        self.funcx_client = FuncXClient(http_timeout=60)

        self.endpoints_statuses = {}

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

    def on_model_broadcast(self) -> List:
        """Sends the model and config to endpoints for FL training.

        Returns
        -------
        tasks: List[future]
            return a list of futures. The futures should support methods .done() and .result()

        """
        # define list storage for results
        tasks = deque()

        logger.debug(f"Launching the {self.executor} executor")
        with self.executor() as executor:
            # submit the corresponding parameters to each endpoint for a round of FL
            for ep, num_s, num_epoch, path_d in zip(
                self.endpoint_ids, self.num_samples, self.epochs, self.path_dir
            ):
                logger.info(f"Starting to broadcast a task to endpoint {ep}")
                try:
                    ep_status = self.funcx_client.get_endpoint_status(ep)["status"]
                except Exception as exp:
                    logger.warning(
                        f"Could not check the status of the endpoint {ep}, the error is: {exp}"
                    )
                    ep_status = "error"

                if ep_status != "online" and self.executor == FuncXExecutor:
                    logger.warning(f"Endpoint {ep} is not online, it's {ep_status}!")
                else:
                    config = self.create_config(num_s, num_epoch, path_d)
                    executor.endpoint_id = ep

                    if self.executor_type == "local":
                        task = executor.submit(
                            self.client_logic.run_round, config, self.model_trainer
                        )

                    elif self.executor_type == "funcx":
                        task = executor.submit(
                            self.client_logic.run_round,
                            self.client_logic,  # funcxExecutor requires the class submitted as well, while the ThreadPoolExecutor does not
                            config,
                            self.model_trainer,
                        )
                    else:
                        raise ValueError(
                            f"{self.executor_type} is invalid executor type, choose one of [local, funcx]"
                        )

                    logger.info(f"Deployed the task to endpoint {ep}")
                    tasks.append(task)

                self.endpoints_statuses[ep] = ep_status

            self.task_start_time = (
                time.time()
            )  # how would this work for multiple endpoints?

            if len(tasks) == 0:
                logger.error(
                    f"The tasks queue is empty, here are the endpoints' statuses: {self.endpoints_statuses}"
                )
                raise ValueError(
                    f"The tasks queue is empty, no tasks were submitted for training!"
                )

        return tasks

    def on_model_receive(self, tasks: List) -> Dict:
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
        endpoint_result_order = []

        logger.info("Starting to retrieve results from endpoints")
        while tasks and (time.time() - self.task_start_time) < self.timeout:
            t = tasks.popleft()
            if t.done():
                res = t.result()
                model_weights.append(res["model_weights"])
                samples_count.append(res["samples_count"])
                endpoint_result_order.append(t)
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
            "endpoint_result_order": endpoint_result_order,
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
        average_weights = np.average(
            results["model_weights"], weights=results["bias_weights"], axis=0
        )
        logger.info("Finished aggregating weights")
        return average_weights

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

    def run_federated_learning(self):
        """The main Federated Learning loop that ties all the functions together.
        Runs <self.rounds> rounds of federated learning"""
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
            logger.info(f"Round {i} evaluation results: ")
            self.on_model_evaluate()
