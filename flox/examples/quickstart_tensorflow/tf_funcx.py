import logging
import os
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from funcx import FuncXExecutor
from tensorflow import keras

from flox.clients.TensorflowClient import TensorflowClient
from flox.controllers.TensorflowController import TensorflowController
from flox.model_trainers.TensorflowTrainer import TensorflowTrainer
from flox.utils import get_test_data

logger = logging.getLogger(__name__)


def main():

    ep1 = "c7487b2b-b129-47e2-989b-5a9ac361befc"

    eps = [ep1]
    logger.info(f"Endpoints: {eps}")

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

    TFTrainer = TensorflowTrainer()
    ClientLogic = TensorflowClient()

    FloxServer = TensorflowController(
        endpoint_ids=eps,
        num_samples=200,
        epochs=2,
        rounds=1,
        client_logic=ClientLogic,
        global_model=global_model,
        executor=FuncXExecutor,  # choose one of [FuncXExecutor, ThreadPoolExecutor]
        executor_type="funcx",  # choose "funcx" for FuncXExecutor, "local" for ThreadPoolExecutor
        model_trainer=TFTrainer,
        path_dir=".",
        x_test=x_test,
        y_test=y_test,
        data_source="keras",
        dataset_name="fashion_mnist",
        preprocess=True,
    )

    logger.info("STARTING FL FLOW...")
    FloxServer.run_federated_learning()


if __name__ == "__main__":
    main()
