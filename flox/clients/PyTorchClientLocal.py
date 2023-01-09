from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from flox.common import EvaluateRes, NDArrays
from flox.logic import BaseModelTrainer, FloxClientLogic


class PyTorchClientLocal(FloxClientLogic):
    def __init__(
        self, model, device=None, criterion=nn.CrossEntropyLoss(), optimizer=None
    ) -> None:
        self.model = model
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.criterion = criterion
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def on_model_receive():
        pass

    def on_data_retrieve(self, config):
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

        # create train DataLoader
        testset = torchvision.datasets.CIFAR10(
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

    def on_model_fit(self, trainloader, config):
        model_weights = self.fit(trainloader, config)

        return model_weights

    def run_round(self, config):
        trainloader, testloader = self.on_data_retrieve(config)
        # fit_results = self.on_model_fit(trainloader, config)
        return {"model_weights": 5, "samples_count": config["num_samples"]}
        # return {"model_weights": fit_results, "samples_count": config["num_samples"]}

    def fit(self, trainloader, config):
        epochs = config["epochs"]
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0

        return self.get_weights()

    def evaluate(self, testloader):
        import torch

        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return EvaluateRes(loss=float(loss), metrics={"accuracy": float(accuracy)})

    def get_weights(self) -> NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: NDArrays) -> None:
        import torch

        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)
