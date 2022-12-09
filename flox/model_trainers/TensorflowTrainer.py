"""TensorFlow ML ModelTrainer Class"""
from tensorflow.keras.models import Sequential

from flox.common import EvaluateRes, NDArrays
from flox.logic import BaseModelTrainer


class TensorflowTrainer(BaseModelTrainer):
    """TensorFlow ML ModelTrainer Class"""

    def __init__(
        self,
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    ) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def get_weights(self, model: Sequential) -> NDArrays:
        import numpy as np

        model_weights = model.get_weights()
        model_weights_numpy = np.asarray(model_weights, dtype=object)
        return model_weights_numpy

    def set_weights(self, model: Sequential, weights: NDArrays) -> None:
        model.set_weights(weights)

    def fit(
        self, model: Sequential, x_train: NDArrays, y_train: NDArrays, epochs: int = 10
    ) -> None:
        model.fit(x_train, y_train, epochs=epochs)

    def evaluate(
        self, model: Sequential, x_test: NDArrays, y_test: NDArrays
    ) -> EvaluateRes:
        scores = model.evaluate(x_test, y_test, verbose=0)
        loss, accuracy = scores[0], scores[1]
        return EvaluateRes(loss=float(loss), metrics={"accuracy": float(accuracy)})

    def get_architecture(self, model: Sequential) -> object:
        """Returns Tensorflow's model architecture as a JSON file"""
        return model.to_json()

    def create_model(self, architecture) -> Sequential:
        """Creates a keras.Sequential model from a JSON architecture file"""
        from tensorflow import keras

        model = keras.models.model_from_json(architecture)
        return model

    def build_model(self, model: Sequential, input_shape) -> None:
        """Builds a keras.Sequential model, requires input_shape. Some older Tensorflow versions
        require the model to be built before we can set new weights"""
        model.build(input_shape=input_shape)

    def compile_model(self, model: Sequential) -> None:
        "Compiles keras.Sequential model"
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
