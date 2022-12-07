import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import Tensor

from flox.clients.TestTorchClient import TestTorchClient
from flox.controllers.TestTorchController import TestTorchController
from flox.model_trainers.PyTorchTrainer import PyTorchTrainer


def get_test_data(tr_split_len=100, te_split_len=100):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    part_tr = torch.utils.data.random_split(
        trainset, [tr_split_len, len(trainset) - tr_split_len]
    )[0]

    trainloader = torch.utils.data.DataLoader(
        part_tr, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    part_te = torch.utils.data.random_split(
        testset, [te_split_len, len(testset) - te_split_len]
    )[0]

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
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
    ClientLogic = TestTorchClient()

    _, testloader = get_test_data()

    ep1 = "bad9c460-0b91-4dac-ba89-7afa5e0e1534"
    eps = [ep1]
    print(f"Endpoints: {eps}")

    FloxServer = TestTorchController(
        endpoint_ids=eps,
        num_samples=100,
        epochs=1,
        rounds=1,
        ClientLogic=ClientLogic,
        ModelTrainer=TorchTrainer,
        path_dir=["."],
        testloader=testloader,
        data_source="keras",
        dataset_name="fashion_mnist",
        preprocess=True,
    )

    print("STARTING FL TORCH FLOW...")
    FloxServer.run_federated_learning()


if __name__ == "__main__":
    main()
