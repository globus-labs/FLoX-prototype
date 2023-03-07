import csv
import os

import numpy as np
from tensorflow import keras


def get_test_data(
    keras_dataset="mnist",
    num_samples=None,
    preprocess=True,
    preprocessing_function=None,
    **kwargs,
):
    """
    Returns (x_test, y_test) of a chosen built-in Keras dataset.
    Also preprocesses the image datasets (mnist, fashion_mnist, cifar10, cifar100) by default.

    Parameters
    ----------
    keras_dataset: str
        one of the available Keras datasets:
        ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing']

    num_samples: int
        randomly samples n data points from (x_test, y_test). Set to None by default.

    preprocess: boolean
        if True, preprocesses (x_test, y_test)

    preprocessing_function: function
        a custom user-provided function that processes (x_test, y_test) and outputs
        a tuple (x_test, y_test)

    Returns
    -------
    (x_test, y_test): tuple of numpy arrays
        testing data

    """
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

    # check if the dataset exists
    if keras_dataset not in available_datasets:
        raise Exception(
            f"Please select one of the built-in Keras datasets: {available_datasets}"
        )

    else:
        _, (x_test, y_test) = dataset_mapping[keras_dataset].load_data()

        # take a random set of images
        if num_samples:
            idx = np.random.choice(np.arange(len(x_test)), num_samples, replace=True)
            x_test = x_test[idx]
            y_test = y_test[idx]

        if preprocess:
            if preprocessing_function and callable(preprocessing_function):
                (x_test, y_test) = preprocessing_function(x_test, y_test)

            else:
                # do default image processing for built-in Keras images
                if keras_dataset in image_datasets:
                    # Scale images to the [0, 1] range
                    x_test = x_test.astype("float32") / 255

                    # Make sure images have shape (num_samples, x, y, 1) if working with MNIST images
                    if x_test.shape[-1] not in [1, 3]:
                        x_test = np.expand_dims(x_test, -1)

                    # convert class vectors to binary class matrices
                    if keras_dataset == "cifar100":
                        num_classes = 100
                    else:
                        num_classes = 10

                    y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_test, y_test)


def create_csv(filename, header):
    if not os.path.exists(filename):
        with open(filename, "a", encoding="UTF8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()


def write_to_csv(filename: str, header: str, data: dict):
    with open(filename, "a", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(data)
