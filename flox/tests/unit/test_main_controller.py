from concurrent.futures import ThreadPoolExecutor

from funcx import FuncXExecutor


def test_init_num_samples_int(tf_controller):
    """MainController.on_model_init() correctly initiates num_samples
    when only a single integer is provided"""
    samples_int = 1000
    tf_controller.num_samples = samples_int
    tf_controller.on_model_init()
    assert tf_controller.num_samples == [1000, 1000, 1000]


def test_init_num_samples_list(tf_controller):
    """MainController.on_model_init() correctly initiates num_samples
    when a list of integers is provided"""

    samples_list = [200, 500, 1000]
    tf_controller.num_samples = samples_list
    tf_controller.on_model_init()
    assert tf_controller.num_samples == samples_list


def test_init_epochs_int(tf_controller):
    """MainController.on_model_init() correctly initiates number of epochs
    when only a single integer is provided"""
    epoch_int = 5
    tf_controller.epochs = epoch_int
    tf_controller.on_model_init()
    assert tf_controller.epochs == [5, 5, 5]


def test_init_epochs_list(tf_controller):
    """MainController.on_model_init() correctly initiates number of epochs
    when a list of integers is provided"""

    epochs_list = [5, 10, 15]
    tf_controller.epochs = epochs_list
    tf_controller.on_model_init()
    assert tf_controller.epochs == epochs_list


def test_init_path_str(tf_controller):
    """MainController.on_model_init() correctly initiates the data directory paths
    when only a single string is provided"""
    path_dir = "."
    tf_controller.path_dir = path_dir
    tf_controller.on_model_init()
    assert tf_controller.path_dir == [".", ".", "."]


def test_init_path_list(tf_controller):
    """MainController.on_model_init() correctly initiates the data directory paths
    when a list of strings is provided"""

    path_dir_list = [".", "./hello/", "./world/"]
    tf_controller.path_dir = path_dir_list
    tf_controller.on_model_init()
    assert tf_controller.path_dir == path_dir_list


def test_init_executor_default(tf_controller):
    """MainController.on_model_init() correctly initiates the executor when the executor
    and executor_type are not provided"""
    tf_controller.executor = None

    tf_controller.on_model_init()

    assert tf_controller.executor_type == "local"
    assert tf_controller.executor == ThreadPoolExecutor


def test_init_executor_provided_type(tf_controller):
    """MainController.on_model_init() correctly initiates the executor when executor_type is provided
    but no executor is provided."""
    tf_controller.executor = None
    tf_controller.executor_type = "funcx"

    tf_controller.on_model_init()

    assert tf_controller.executor_type == "funcx"
    assert tf_controller.executor == FuncXExecutor


def test_init_executor_provided_executor(tf_controller):
    """MainController.on_model_init() correctly initiates the executor when a custom executor is provided."""
    tf_controller.executor = "fake_executor"
    tf_controller.executor_type = "local"

    tf_controller.on_model_init()

    assert tf_controller.executor_type == "local"
    assert tf_controller.executor == "fake_executor"
