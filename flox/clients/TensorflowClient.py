import os

import numpy as np
from tensorflow import keras

from flox.clients.MainClient import MainClient


class TensorflowClient(MainClient):
    def retrieve_framework_data(self, config):
        dataset_mapping = {
            "mnist": keras.datasets.mnist,
            "fashion_mnist": keras.datasets.fashion_mnist,
            "cifar10": keras.datasets.cifar10,
            "cifar100": keras.datasets.cifar100,
            "imdb": keras.datasets.imdb,
            "reuters": keras.datasets.reuters,
            "boston_housing": keras.datasets.boston_housing,
        }
        available_datasets = dataset_mapping.keys()

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

        return (x_train, y_train)

    def on_data_process(self, training_data, config):
        x_train, y_train = training_data
        if config["preprocess"]:
            if config["data_source"] == "framework":
                # do default image processing for built-in Keras images
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

            elif config["data_source"] == "local":
                depth = config["input_shape"][3]
                image_size_y = config["input_shape"][2]
                image_size_x = config["input_shape"][1]

                # take a limited number of samples, if indicated
                if config["num_samples"]:
                    idx = np.random.choice(
                        np.arange(len(x_train)), config["num_samples"], replace=True
                    )
                    x_train = x_train[idx]
                    y_train = y_train[idx]

                # reshape and scale to pixel values to 0-1
                x_train = x_train.reshape(
                    len(x_train), image_size_x, image_size_y, depth
                )
                x_train = x_train / 255.0

            else:
                raise KeyError("Please choose one of data sources: ['local', 'keras']")

        return (x_train, y_train)

    def on_model_fit(self, model_trainer, config, training_data):
        import numpy as np

        x_train, y_train = training_data
        model = model_trainer.create_model(config["architecture"])
        model_trainer.compile_model(model)
        model_trainer.set_weights(model, config["weights"])

        model_trainer.fit(
            model,
            x_train,
            y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
        )
        model_weights = model_trainer.get_weights(model)

        # transform to a numpy array
        np_model_weights = np.asarray(model_weights, dtype=object)

        return np_model_weights

    # def on_model_send(self, fit_results, training_data, config=None):
    #     x_train, y_train = training_data
    #     return {"model_weights": fit_results,
    #             "samples_count": x_train.shape[0]}

    def get_number_of_samples(self, training_data, config):
        x_train, _ = training_data
        return x_train.shape[0]

    # def run_round(self, config, model_trainer):
    #     x_train, y_train = self.on_data_retrieve(config)

    #     fit_results = self.on_model_fit(model_trainer, config, x_train, y_train)

    #     # x_train.shape[0] - task_n_samples
    #     return {"model_weights": fit_results, "samples_count": x_train.shape[0]}
