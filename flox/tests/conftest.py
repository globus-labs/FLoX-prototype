import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from funcx import FuncXExecutor

from flox.clients.PyTorchClient import PyTorchClient
from flox.clients.TensorflowClient import TensorflowClient
from flox.controllers.PyTorchController import PyTorchController
from flox.controllers.TensorflowController import TensorflowController
from flox.model_trainers.PyTorchTrainer import PyTorchTrainer
from flox.model_trainers.TensorflowTrainer import TensorflowTrainer
from flox.utils import get_test_data


@pytest.fixture
def tf_controller():
    # `fashion_mnist` images are grayscale, 28 x 28 pixels in size
    input_shape = (28, 28, 1)
    # there are 10 classes in the dataset
    num_classes = 10

    # define the model architecture
    global_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # compile the model
    global_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    x_test, y_test = get_test_data(keras_dataset="fashion_mnist", num_samples=2000)

    tf_trainer = TensorflowTrainer()
    tf_client_logic = TensorflowClient()

    eps = ["fake_endpoint_1", "fake_endpoint_2", "fake_endpoint_3"]

    tf_controller = TensorflowController(
        endpoint_ids=eps,
        num_samples=200,
        epochs=2,
        rounds=2,
        client_logic=tf_client_logic,
        global_model=global_model,
        executor_type="local",  # choose "funcx" for FuncXExecutor, "local" for ThreadPoolExecutor
        model_trainer=tf_trainer,
        path_dir=".",
        x_test=x_test,
        y_test=y_test,
        data_source="keras",
        dataset_name="fashion_mnist",
        preprocess=True,
    )

    return tf_controller


@pytest.fixture
def pytorch_controller():
    def get_test_data(config):
        import torch
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        batch_size = config.get("batch_size", 32)
        dataset_name = config["dataset_name"]
        num_workers = config.get("num_workers", 8)
        root = config.get("data_root", "./data")

        # create train DataLoader
        trainset = dataset_name(
            root=root, train=True, download=True, transform=transform
        )

        train_split_len = (
            len(trainset)
            if "num_samples" not in config.keys()
            else config["num_samples"]
        )

        train_subpart = torch.utils.data.random_split(
            trainset, [train_split_len, len(trainset) - train_split_len]
        )[0]
        trainloader = torch.utils.data.DataLoader(
            train_subpart, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # create test DataLoader
        testset = dataset_name(
            root=root, train=False, download=True, transform=transform
        )
        test_split_len = (
            len(trainset)
            if "num_samples" not in config.keys()
            else config["num_samples"]
        )

        test_subpart = torch.utils.data.random_split(
            testset, [test_split_len, len(testset) - test_split_len]
        )[0]
        testloader = torch.utils.data.DataLoader(
            test_subpart, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return trainloader, testloader

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            import torch

            x = self.pool(torch.nn.functional.relu(self.conv1(x)))
            x = self.pool(torch.nn.functional.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    torch_trainer = PyTorchTrainer(net)
    torch_client = PyTorchClient()

    data_config = {
        "num_samples": 1000,
        "batch_size": 32,
        "dataset_name": torchvision.datasets.CIFAR10,
        "num_workers": 4,
    }

    _, testloader = get_test_data(data_config)

    eps = ["fake_endpoint_1", "fake_endpoint_2", "fake_endpoint_3"]

    flox_controller = PyTorchController(
        endpoint_ids=eps,
        num_samples=100,
        epochs=1,
        rounds=2,
        client_logic=torch_client,
        model_trainer=torch_trainer,
        executor_type="local",  # choose "funcx" for FuncXExecutor, "local" for ThreadPoolExecutor
        testloader=testloader,
        dataset_name=torchvision.datasets.CIFAR10,
    )

    return flox_controller
