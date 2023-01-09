"""PyTorch ML ModelTrainer Class"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from flox.common import EvaluateRes, NDArrays
from flox.logic import BaseModelTrainer


class PyTorchTrainerLocal(BaseModelTrainer):
    """PyTorch ML ModelTrainer Class"""

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

    # def create_dataloader(self, x_data, y_data, batch_size=4, shuffle=True) -> torch.utils.data.DataLoader:
    #     combined_data = torch.hstack((x_data, y_data))
    #     data_loader = torch.utils.data.DataLoader(combined_data, batch_size=batch_size, shuffle=shuffle)
    #     return data_loader
