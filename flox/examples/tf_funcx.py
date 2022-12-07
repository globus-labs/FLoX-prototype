import os

import tensorflow as tf
from tensorflow import keras

from flox.clients.TestTensorflowClient import TestTensorflowClient
from flox.controllers.TestTensorflowController import TestTensorflowController
from flox.model_trainers.TensorflowTrainer import TensorflowTrainer
from flox.utils import get_test_data


def main():

    # ep1 = "fe49ba41-9654-4d1b-8266-fd2f8197b242"
    ep1 = "ef588445-ce77-4839-ac9b-646465f872ee"
    eps = [ep1]
    print(f"Endpoints: {eps}")

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
    ClientLogic = TestTensorflowClient()

    FloxServer = TestTensorflowController(
        endpoint_ids=eps,
        num_samples=500,
        epochs=5,
        rounds=7,
        ClientLogic=ClientLogic,
        global_model=global_model,
        ModelTrainer=TFTrainer,
        path_dir=["."],
        x_test=x_test,
        y_test=y_test,
        data_source="keras",
        dataset_name="fashion_mnist",
        preprocess=True,
    )

    print("STARTING FL FLOW...")
    FloxServer.run_federated_learning()


if __name__ == "__main__":
    main()
