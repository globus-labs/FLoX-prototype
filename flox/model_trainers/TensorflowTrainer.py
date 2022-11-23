import numpy as np
import tensorflow as tf
from tensorflow import keras

import flox
from flox.common import EvaluateRes, NDArray, NDArrays


class TensorflowTrainer(flox.logic.base_model_trainer.BaseModelTrainer):
    def __init__(
        self,
        model,
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        epochs=10,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        # test data too?

    def get_weights(self) -> NDArrays:
        model_weights = self.model.get_weights()
        model_weights_numpy = np.asarray(model_weights, dtype=object)
        return model_weights_numpy

    def set_weights(self, weights: NDArrays) -> None:
        self.model.set_weights(weights)

    def get_architecture(self):
        return self.model.to_json()

    def create_model(self, architecture) -> None:
        self.model = keras.models.model_from_json(architecture)

    def compile_model(self) -> None:
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )

    def fit(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train, epochs=self.epochs)

    def evaluate(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        loss, accuracy = scores[0], scores[1]
        return EvaluateRes(loss=float(loss), metrics={"accuracy": float(accuracy)})
