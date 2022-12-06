# from tensorflow import keras
from flox.common import EvaluateRes, NDArray, NDArrays
from flox.logic import BaseModelTrainer


class TensorflowTrainer(BaseModelTrainer):
    def __init__(
        self,
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    ) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def get_weights(self, model) -> NDArrays:
        import numpy as np

        model_weights = model.get_weights()
        model_weights_numpy = np.asarray(model_weights, dtype=object)
        return model_weights_numpy

    def set_weights(self, model, weights: NDArrays) -> None:
        model.set_weights(weights)

    def get_architecture(self, model):
        return model.to_json()

    def create_model(self, architecture) -> None:
        from tensorflow import keras

        model = keras.models.model_from_json(architecture)
        return model

    def build_model(self, model, input_shape) -> None:
        model.build(input_shape=input_shape)

    def compile_model(self, model) -> None:
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def fit(self, model, x_train, y_train, epochs=10) -> None:
        model.fit(x_train, y_train, epochs=epochs)

    def evaluate(self, model, x_test, y_test):
        scores = model.evaluate(x_test, y_test, verbose=0)
        loss, accuracy = scores[0], scores[1]
        return EvaluateRes(loss=float(loss), metrics={"accuracy": float(accuracy)})
