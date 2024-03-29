import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from funcx import FuncXExecutor
from torch import Tensor

from flox.clients.PyTorchClient import PyTorchClient
from flox.controllers.PyTorchController import PyTorchController
from flox.model_trainers.PyTorchTrainer import PyTorchTrainer

logger = logging.getLogger(__name__)


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
    trainset = dataset_name(root=root, train=True, download=True, transform=transform)

    train_split_len = (
        len(trainset) if "num_samples" not in config.keys() else config["num_samples"]
    )

    train_subpart = torch.utils.data.random_split(
        trainset, [train_split_len, len(trainset) - train_split_len]
    )[0]
    trainloader = torch.utils.data.DataLoader(
        train_subpart, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # create test DataLoader
    testset = dataset_name(root=root, train=False, download=True, transform=transform)
    test_split_len = (
        len(trainset) if "num_samples" not in config.keys() else config["num_samples"]
    )

    test_subpart = torch.utils.data.random_split(
        testset, [test_split_len, len(testset) - test_split_len]
    )[0]
    testloader = torch.utils.data.DataLoader(
        test_subpart, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def main():
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

    ep1 = "fb93a1c2-a8d7-49f3-ad59-375f4e298784"

    eps = [ep1, "fake1", "fake2"]
    logger.info(f"Endpoints: {eps}")

    flox_controller = PyTorchController(
        endpoint_ids=eps,
        num_samples=100,
        epochs=1,
        rounds=3,
        client_logic=torch_client,
        model_trainer=torch_trainer,
        executor_type="local",  # choose "funcx" for FuncXExecutor, "local" for ThreadPoolExecutor
        path_dir=".",
        data_source="framework",
        testloader=testloader,
        dataset_name=torchvision.datasets.CIFAR10,
        preprocess=True,
        tasks_per_endpoint=1,
        csv_filename="test_evaluation_3.csv",
    )

    logger.info("STARTING FL TORCH FLOW...")
    flox_controller.run_federated_learning()


if __name__ == "__main__":
    main()
