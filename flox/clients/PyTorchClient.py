from flox.clients.MainClient import MainClient


class PyTorchClient(MainClient):
    def retrieve_framework_data(self, config):
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

        return trainloader

    def on_model_fit(self, model_trainer, config, training_data):
        model_weights = model_trainer.fit(training_data, config)

        return model_weights

    # def run_round(self, config, model_trainer):
    #     trainloader = self.on_data_retrieve(config)
    #     fit_results = self.on_model_fit(model_trainer, trainloader, config)

    #     return {"model_weights": fit_results, "samples_count": config["num_samples"]}
