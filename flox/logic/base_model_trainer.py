from flox.common import NDArrays


class BaseModelTrainer:
    """Abstract base class for FLoX ML model trainers"""

    def fit(self) -> NDArrays:
        """Fits the model using training data

        Returns
        ----------
        NDArrays
            the new model weights in the Numpy array form

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def evaluate(self):
        """Evaluates the model using testing data"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def get_weights(self) -> NDArrays:
        """Returns the weights of the model as a Numpy array


        Returns
        -------
        NDArrays
            the model's weights as Numpy arrays

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def set_weights(self, new_weights: NDArrays) -> None:
        """Sets the model's weights to the new weights

        Parameters
        ----------
        new_weights: NDArrays
            new model weights in the Numpy array form

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")
