from funcx import FuncXExecutor


def test_keras(config):
    import os

    import numpy as np
    from tensorflow import keras

    if config["data_source"] == "keras":
        available_datasets = [
            "mnist",
            "fashion_mnist",
            "cifar10",
            "cifar100",
            "imdb",
            "reuters",
            "boston_housing",
        ]
        dataset_mapping = {
            "mnist": keras.datasets.mnist,
            "fashion_mnist": keras.datasets.fashion_mnist,
            "cifar10": keras.datasets.cifar10,
            "cifar100": keras.datasets.cifar100,
            "imdb": keras.datasets.imdb,
            "reuters": keras.datasets.reuters,
            "boston_housing": keras.datasets.boston_housing,
        }
        image_datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

        # check if the Keras dataset exists
        if config["dataset_name"] not in available_datasets:
            raise Exception(
                f"Please select one of the built-in Keras datasets: {available_datasets}"
            )

        else:
            # load the data
            (x_train, y_train), _ = dataset_mapping[config["dataset_name"]].load_data()

            # take a random set of images
            if config["num_samples"]:
                idx = np.random.choice(
                    np.arange(len(x_train)), config["num_samples"], replace=True
                )
                x_train = x_train[idx]
                y_train = y_train[idx]

            if config["preprocess"]:
                # do default image processing for built-in Keras images
                if config["dataset_name"] in image_datasets:
                    # Scale images to the [0, 1] range
                    x_train = x_train.astype("float32") / 255

                    # Make sure images have shape (num_samples, x, y, 1) if working with MNIST images
                    if x_train.shape[-1] not in [1, 3]:
                        x_train = np.expand_dims(x_train, -1)

                    # convert class vectors to binary class matrices
                    if config["dataset_name"] == "cifar100":
                        num_classes = 100
                    else:
                        num_classes = 10

                    y_train = keras.utils.to_categorical(y_train, num_classes)

    else:
        raise Exception("Please choose one of data sources: ['local', 'keras']")

    return "SUCCESSFULLY RETRIEVED YOUR DATA"


def main():

    ep1 = "ef588445-ce77-4839-ac9b-646465f872ee"

    config = {}
    config["dataset_name"] = "fashion_mnist"
    config["num_samples"] = 200
    config["preprocess"] = True
    config["data_source"] = "keras"

    with FuncXExecutor(endpoint_id=ep1) as fx:
        task = fx.submit(test_keras, config)

    print(task.result())


if __name__ == "__main__":
    main()
