import tensorflow as tf
from tensorflow import keras

from flox.utils import get_test_data


class TensorflowTrainer:
    def __init__(
        self,
        model,
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def get_weights(self):
        import numpy as np

        model_weights = self.model.get_weights()
        model_weights_numpy = np.asarray(model_weights, dtype=object)
        return model_weights_numpy

    def set_weights(self, weights) -> None:
        self.model.set_weights(weights)

    def get_architecture(self):
        return self.model.to_json()

    def create_model(self, architecture) -> None:
        self.model = keras.models.model_from_json(architecture)

    def build_model(self, input_shape) -> None:
        self.model.build(input_shape=input_shape)

    def compile_model(self) -> None:
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

    def fit(self, x_train, y_train, epochs=10) -> None:
        self.model.fit(x_train, y_train, epochs=epochs)

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        loss, accuracy = scores[0], scores[1]
        return {"loss": loss, "accuracy": accuracy}


import numpy as np
from funcx import FuncXExecutor


class TestServer:
    def __init__(
        self,
        endpoint_ids=None,
        num_samples=None,
        epochs=None,
        rounds=None,
        ClientLogic=None,
        ModelTrainer=None,
        path_dir=None,
        x_test=None,
        y_test=None,
        data_source=None,
        dataset_name=None,
        preprocess=None,
    ):
        self.endpoint_ids = endpoint_ids
        self.num_samples = num_samples
        self.epochs = epochs
        self.rounds = rounds
        self.ClientLogic = ClientLogic
        self.ModelTrainer = ModelTrainer
        self.path_dir = path_dir
        self.x_test = x_test
        self.y_test = y_test
        self.data_source = data_source
        self.dataset_name = dataset_name
        self.preprocess = preprocess

    def on_model_init(self):
        # if num_samples or epochs is an int, convert to list so the same number can be applied to all endpoints
        if type(self.num_samples) == int:
            self.num_samples = [self.num_samples] * len(self.endpoint_ids)

        if type(self.epochs) == int:
            self.epochs = [self.epochs] * len(self.endpoint_ids)

    def on_model_broadcast(self):
        """DocString"""
        # get the model's architecture
        model_architecture = self.ModelTrainer.get_architecture()

        # define list storage for results
        tasks = []

        # submit the corresponding parameters to each endpoint for a round of FL
        for ep, num_s, num_epoch, path_d in zip(
            self.endpoint_ids, self.num_samples, self.epochs, self.path_dir
        ):
            config = {
                "num_samples": num_s,
                "epochs": num_epoch,
                "path_dir": path_d,
                "data_source": self.data_source,
                "dataset_name": self.dataset_name,
                "preprocess": self.preprocess,
            }
            with FuncXExecutor(endpoint_id=ep) as fx:
                task = fx.submit(self.ClientLogic.run_round, config, self.ModelTrainer)

                tasks.append(task)
        return tasks

    def on_model_receive(self, tasks):
        """DocString"""
        # extract model updates from each endpoints once they are available
        model_weights = [t.result()["model_weights"] for t in tasks]
        samples_count = np.array([t.result()["samples_count"] for t in tasks])

        total = sum(samples_count)
        fractions = samples_count / total

        return {
            "model_weights": model_weights,
            "samples_count": samples_count,
            "bias_weights": fractions,
        }

    def on_model_aggregate(self, results):
        """DocString"""
        average_weights = np.average(
            results["model_weights"], weights=results["bias_weights"], axis=0
        )

        return average_weights

    def on_model_update(self, updated_weights) -> None:
        """DocString"""
        self.ModelTrainer.set_weights(updated_weights)

    def on_model_evaluate(self, x_test, y_test):
        results = self.ModelTrainer.evaluate(x_test, y_test)
        print(results)

    def run_federated_learning(self):
        self.on_model_init()

        # start running FL loops
        for i in range(self.rounds):
            # broadcast the model
            tasks = self.on_model_broadcast()

            # process & decrypt the results
            results = self.on_model_receive(tasks)
            # aggregate the weights
            updated_weights = self.on_model_aggregate(results)

            # update the model's weights
            self.on_model_update(updated_weights)

            # evaluate the model
            print(f"Round {i} evaluation results: ")
            self.on_model_evaluate(self.x_test, self.y_test)


class TestClient:
    def on_model_receive():
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

    def on_data_prepare():
        """DocString"""
        pass

    def on_model_fit(self, ModelTrainer, config, x_train, y_train):
        """DocString"""
        import numpy as np

        # ModelTrainer = config['ModelTrainer']

        ModelTrainer.fit(x_train, y_train, epochs=config["epochs"])
        model_weights = ModelTrainer.get_weights()

        # transform to a numpy array
        np_model_weights = np.asarray(model_weights, dtype=object)

        return np_model_weights

    def on_model_send():
        """DocString"""
        pass

    def run_round(self, config, ModelTrainer):
        """DocString"""

        x_train, y_train = self.on_data_retrieve(config)

        fit_results = self.on_model_fit(ModelTrainer, config, x_train, y_train)

        return {"model_weights": fit_results, "samples_count": x_train.shape[0]}


ep1 = "98279cc2-ce58-4e3c-a6b9-2f237a46fbe0"
eps = [ep1]

# `fashion_mnist` images are grayscale, 28 x 28 pixels in size
input_shape = (28, 28, 1)
# there are 10 classes in the dataset
num_classes = 10

# define the model architecture
global_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# compile the model
global_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
x_test, y_test = get_test_data(keras_dataset="fashion_mnist", num_samples=2000)

TFTrainer = TensorflowTrainer(global_model)
ClientLogic = TestClient()

FloxServer = TestServer(
    endpoint_ids=eps,
    num_samples=100,
    epochs=5,
    rounds=2,
    ClientLogic=ClientLogic,
    ModelTrainer=TFTrainer,
    path_dir=["."],
    x_test=x_test,
    y_test=y_test,
    data_source="keras",
    dataset_name="fashion_mnist",
    preprocess=True,
)

print("starting the fl flow")
FloxServer.run_federated_learning()
