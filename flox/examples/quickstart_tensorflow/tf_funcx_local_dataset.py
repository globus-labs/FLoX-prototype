import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

from flox.clients.TensorflowClient import TensorflowClient
from flox.controllers.TensorflowController import TensorflowController
from flox.model_trainers.TensorflowTrainer import TensorflowTrainer
from flox.utils import get_test_data


def process_data(train_image, train_label, num_samples=None):
    depth = 3
    image_size_y = 32
    image_size_x = 32

    if num_samples:
        idx = np.random.choice(np.arange(len(train_image)), num_samples, replace=True)
        train_image = train_image[idx]
        train_label = train_label[idx]

    train_image = train_image.reshape(
        len(train_image), image_size_x, image_size_y, depth
    )
    train_image = train_image / 255.0

    return (train_image, train_label)


def main():

    # ep1 = "fe49ba41-9654-4d1b-8266-fd2f8197b242"
    ep1 = "68e4332b-e0dd-4933-a2f9-a3d7039764f6"
    eps = [ep1]
    print(f"Endpoints: {eps}")

    # `fashion_mnist` images are grayscale, 28 x 28 pixels in size
    input_shape = (32, 32, 3)
    # there are 10 classes in the dataset
    num_classes = 10

    # define the model architecture
    global_model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # compile the model
    global_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    with open("data/test_data_animal10_32.pkl", "rb") as file:
        x_test, y_test = pickle.load(file)

    x_test, y_test = process_data(x_test, y_test)

    TFTrainer = TensorflowTrainer(loss="sparse_categorical_crossentropy")
    ClientLogic = TensorflowClient()

    FloxServer = TensorflowController(
        endpoint_ids=eps,
        num_samples=500,
        epochs=5,
        rounds=3,
        client_logic=ClientLogic,
        global_model=global_model,
        model_trainer=TFTrainer,
        data_source="local",
        path_dir="/home/pi/datasets",
        x_train_filename="x_animal10_32.npy",
        y_train_filename="y_animal10_32.npy",
        input_shape=(32, 32, 32, 3),
        preprocess=True,
        x_test=x_test,
        y_test=y_test,
    )

    print("STARTING FL FLOW...")
    FloxServer.run_federated_learning()


if __name__ == "__main__":
    main()
