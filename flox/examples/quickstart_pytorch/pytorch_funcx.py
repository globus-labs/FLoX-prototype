import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import Tensor

from flox.clients.PyTorchClient import PyTorchClient
from flox.controllers.PyTorchController import PyTorchController
from flox.model_trainers.PyTorchTrainer import PyTorchTrainer


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

    batch_size = 32 if "batch_size" not in config.keys() else config["batch_size"]
    dataset_name = config["dataset_name"]
    num_workers = 8 if "num_workers" not in config.keys() else config["num_workers"]
    root = "./data" if "data_root" not in config.keys() else config["data_root"]

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

    # create train DataLoader
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
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
    TorchTrainer = PyTorchTrainer(net)
    ClientLogic = PyTorchClient()

    data_config = {
        "num_samples": 1000,
        "batch_size": 32,
        "dataset_name": torchvision.datasets.FashionMNIST,
        "num_workers": 4,
    }

    _, testloader = get_test_data(data_config)

    ep1 = "bad9c460-0b91-4dac-ba89-7afa5e0e1534"
    eps = [ep1]
    print(f"Endpoints: {eps}")

    FloxServer = PyTorchController(
        endpoint_ids=eps,
        num_samples=100,
        epochs=1,
        rounds=1,
        ClientLogic=ClientLogic,
        ModelTrainer=TorchTrainer,
        path_dir=["."],
        testloader=testloader,
        dataset_name=torchvision.datasets.FashionMNIST,
        preprocess=True,
    )

    print("STARTING FL TORCH FLOW...")
    FloxServer.run_federated_learning()


if __name__ == "__main__":
    main()
