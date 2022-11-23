from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import Tensor

import flox
from flox.common import EvaluateRes, NDArray, NDArrays


class PyTorchTrainer(flox.logic.base_model_trainer.BaseModelTrainer):
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), optimizer=None) -> None:
        self.model = model
        self.criterion = criterion
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def fit(self, trainloader, device, epochs=10):
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                images, labels = data[0].to(device), data[1].to(device)

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

    def evaluate(self, testloader, device):
        """DocString"""
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
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
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def create_model(self):
        """DocString"""
        pass

    def compile_model(self):
        """DocString"""
        pass

    def get_architecture(self):
        """DocString"""
        pass

    # def create_dataloader(self, x_data, y_data, batch_size=4, shuffle=True) -> torch.utils.data.DataLoader:
    #     combined_data = torch.hstack((x_data, y_data))
    #     data_loader = torch.utils.data.DataLoader(combined_data, batch_size=batch_size, shuffle=shuffle)
    #     return data_loader
