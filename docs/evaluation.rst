.. _evaluation:

----------
Evaluation
----------

Evaluation is built-in into FLoX and can be enabled by passing a '.csv' filename path to the *Controller*:

.. code-block:: python

    flox_controller = TensorflowController(
        ...
        csv_filename="evaluation_data.csv",
        evaluate_individual_models=True
        ...
    )

If ``evaluate_individual_models`` is ``True``, FLoX will also get the loss and accuracy of every
individual model what was trained on endpoints (note: currently individual model evaluation is
only available for Tensorflow models. Support for PyTorch will be added too, and PRs implementing
that are welcome).


Important Definitions:
---------------------
* "Endpoint" - borrowing from `funcx <https://funcx.readthedocs.io/en/latest/endpoints.html>`_,
an endpoint is a persistent service launched
by the user on a compute system to serve as a conduit for executing functions on that computer.
You can have multiple endpoints on the same physical device.

* "Task" - task is an instance of a function invocation and its result. FLoX packages the
FL loop into a function that is then invoked on each provided endpoint. If you want to make
use of multiple cores on your devices, FLoX can spin off multiple tasks on the same endpoint
at the same time, using threading and multiprocessing.

* "Round" - round is a single iteration of the Federated Learning loop consisting of model broadcasting,
model training, aggregation of weights, and evaluation (optional).

* "Experiment" - experiment consists of one or multiple rounds of Federated Learning.
When you start ``.run_federated_learning``, you have started an experiment that will last
either until all FL rounds have been completed, an error has occured, or you manually stopped the experiment.

Recorded Attributes
-------------------
As of now, these are the attributes that are being recorded by FLoX:

* "experiment_id" - a user-provided id or an automatically assigned uuid for tracking the experiments

* "experiment_name" - a user-provided name for the experiment (optional)

* "experiment_description" - a user-provided description for the experiment (optional)

* "experiment_executor" - executor that is used to facilitate execution of FL operations on the endpoints

* "dataset_name" - dataset name

* "data_source" - source of the data for FL training ("local"/"framework"). "local" means that the data was retrieved from a file stored locally. "framework" means that the data was retrieved from one of the default datasets such as ``keras.datasets.mnist``)

* "round_number" - the sequence of the FL round starting from index 0

* "n_clients_provided" - how many endpoints the user provided in the ``endpoind_ids`` argument. The number of clients/endpoints that actually participated in the training can differ if the endpoints were not available

* "n_tasks_retrieved" - how many tasks were completed during the corresponding round

* "round_aggregated_accuracy" - accuracy of the aggregated model during the corresponding round, as per evaluation on the provided test data.

* "round_aggregated_loss" - loss of the aggregated model during the corresponding round, as per evaluation on the provided test data.

* "round_aggregation_runtime" - runtime of aggregation during the corresponding round

* "running_average_aggregation" - if the attribute ``running_average`` was set to ``True``, this will store the total aggregation runtime during the corresponding round. See the docstring for ``MainController.on_model_receive()`` to learn more about running average aggregation.

* "total_round_runtime" - the total runtime of the round, *excluding* evaluation.

* "round_start_time" - UTC timestamp of when the round started

* "round_end_time" - UTC timestamp of when the round ended

* "endpoint_uuid" - uuid of the endpoint as provided by the user

* "endpoint_custom_name" - user-provided endpoint name to help with tracking endpoints

* "endpoint_latest_status" - latest endpoint status ("online"/"offline"/"error"). Only applicable when using the ``funcx`` executor.

* "endpoint_ram" - ram of the endpoint as returned by ``psutil.virtual_memory().total``

* "endpoint_physical_cores" - number of physical cores as returned by ``psutil.cpu_count(logical=False)``

* "endpoint_logical_cores" - number of logical cores as returned by ``psutil.cpu_count(logical=True)``

* "endpoint_platform_name" - endpoint platform name as returned by ``platform.node()``

* "number_of_tasks" - number of tasks submitted to the corresponding endpoint

* "desired_n_samples" - desired number of samples for the task. Sometimes there may not be as many samples as the user indicated, so the actual number of samples used in training may differ

* "epochs" - desired number of samples for training.

* "batch_size" - desired batch_size for training.

* "task_local_uuid" - an automatically assigned UUID to the task used for internal task tracking

* "task_funcx_uuid" - a UUID assigned by the ``funcX`` service when the task is submitted to the endpoint if using "local" executor type, this will equate for NaN

* "file_size" - size of the model being transferred to the endpoint

* "task_actual_n_samples" - actual number of samples used for training during the corresponding task

* "task_model_accuracy" - accuracy of the individual model returned from the task

* "task_model_loss" - loss of the individual model returned from the task

* "task_runtime" - total runtime of the task, measured on the endpoint's end

* "task_training_runtime" - time it took to train the model on the endpoint

* "task_data_processing_runtime" - time it took to retrieve and process data on the endpoint

* "task_broadcasted_to_retrieved_runtime" - the time it took from broadcasting the task to retrieving results of the task, measured on the *Controller*

* "task_broadcast_start_time" - UTC timestamp of when the task broadcasting started, taken on the *Controller*

* "task_broadcast_finish_time"  UTC timestamp of when the task broadcasting ended, taken on the *Controller*

* "task_start_time" - UTC timestamp of when the task started, taken on the *Client*

* "task_finish_time" - UTC timestamp of when the task finished, taken on the *Client*

* "task_results_received_time" - UTC timestamp of when the task results were retrieved, taken on the *Controller*
