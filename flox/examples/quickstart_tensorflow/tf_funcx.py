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

    ep1 = "62f1e8da-efa0-4474-be4d-fe298234fcc9"
    ep2 = "e03cac12-8f8f-4bf2-bf03-0378da93ff2c"

    eps = [ep1, ep2]
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

    tf_trainer = TensorflowTrainer()
    tf_client = TensorflowClient()

    flox_controller = TensorflowController(
        endpoint_ids=eps,
        num_samples=200,
        epochs=2,
        rounds=3,
        client_logic=tf_client,
        global_model=global_model,
        executor_type="funcx",  # choose "funcx" for FuncXExecutor, "local" for ThreadPoolExecutor
        model_trainer=tf_trainer,
        x_test=x_test,
        y_test=y_test,
        data_source="keras",
        dataset_name="fashion_mnist",
        preprocess=True,
        tasks_per_endpoint=4,
    )

    logger.info("STARTING FL FLOW...")
    flox_controller.run_federated_learning()


if __name__ == "__main__":
    main()
