from flox.logic import FloxClientLogic


class TestTorchClient(FloxClientLogic):
    def on_model_receive():
        pass

    def on_data_retrieve(self, config):
        import torch
        import torchvision
        import torchvision.transforms as transforms

        tr_split_len = te_split_len = config["num_samples"]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
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

    def on_model_fit(self, ModelTrainer, trainloader, config):
        """DocString"""
        model_weights = ModelTrainer.fit(trainloader, config)

        return model_weights

    def run_round(self, config, ModelTrainer):
        """DocString"""

        trainloader, testloader = self.on_data_retrieve(config)
        fit_results = self.on_model_fit(ModelTrainer, trainloader, config)

        return {"model_weights": fit_results, "samples_count": config["num_samples"]}
