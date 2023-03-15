import logging
import os

import tensorflow as tf
from funcx import FuncXExecutor
from tensorflow import keras

from flox.clients.TensorflowClient import TensorflowClient
from flox.controllers.TensorflowController import TensorflowController
from flox.model_trainers.TensorflowTrainer import TensorflowTrainer
from flox.utils import get_test_data

logger = logging.getLogger(__name__)


def main():

    ep1 = "7a9e7979-b88c-4494-9636-16970b7199e8"
    ep2 = "02cb5f88-9313-423f-9f5e-02dd7f8a4137"
    ep3 = "aed3a876-d022-42d5-8f7f-bb1ad75266f4"

    eps = [ep1, ep2, ep3]
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
        num_samples=[100, 200, 300],
        epochs=[2, 3, 4],
        rounds=1,
        client_logic=tf_client,
        global_model=global_model,
        executor_type="funcx",  # choose "funcx" for FuncXExecutor, "local" for ThreadPoolExecutor
        model_trainer=tf_trainer,
        x_test=x_test,
        y_test=y_test,
        data_source="framework",
        dataset_name="fashion_mnist",
        preprocess=True,
        tasks_per_endpoint=[1, 3, 2],
        csv_filename="test_evaluation_3.csv",
        evaluate_individual_models=True,
        running_average=True,
    )

    logger.info("STARTING FL FLOW...")
    flox_controller.run_federated_learning()


if __name__ == "__main__":
    main()
