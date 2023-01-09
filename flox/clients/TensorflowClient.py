from flox.logic import FloxClientLogic


class TensorflowClient(FloxClientLogic):
    def on_model_receive(self):
        pass

    def on_data_retrieve(self, config):
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
                (x_train, y_train), _ = dataset_mapping[
                    config["dataset_name"]
                ].load_data()

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

        return x_train, y_train

    def on_model_fit(self, ModelTrainer, config, x_train, y_train):
        import numpy as np

        model = ModelTrainer.create_model(config["architecture"])
        ModelTrainer.compile_model(model)
        ModelTrainer.set_weights(model, config["weights"])

        ModelTrainer.fit(model, x_train, y_train, epochs=config["epochs"])
        model_weights = ModelTrainer.get_weights(model)

        # transform to a numpy array
        np_model_weights = np.asarray(model_weights, dtype=object)

        return np_model_weights

    def on_model_send(self):
        pass

    def run_round(self, config, ModelTrainer):
        x_train, y_train = self.on_data_retrieve(config)

        fit_results = self.on_model_fit(ModelTrainer, config, x_train, y_train)

        return {"model_weights": fit_results, "samples_count": x_train.shape[0]}
