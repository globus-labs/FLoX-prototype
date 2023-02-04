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
