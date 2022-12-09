def on_data_retrieve(config):
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
    import torchvision

    config = {}
    # config['num_samples'] = 200
    config["dataset_name"] = torchvision.datasets.CIFAR10
    # config['batch_size'] = 32
    # config['num_workers'] = 4

    train, test = on_data_retrieve(config)

    print(train, test, len(train), len(test))


if __name__ == "__main__":
    main()
