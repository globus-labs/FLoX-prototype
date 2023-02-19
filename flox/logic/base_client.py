"""Abstract Base Class for FLoX Clients"""
from flox.common import NDArrays, XYData


class FloxClientLogic:
    """Abstract base class for FLoX Clients"""

    def on_model_receive(self) -> None:
        """Parses & decrypts the received data from a controller"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_data_retrieve(self, config: dict) -> XYData:
        """Retrieves data for training the model.

        Parameters
        ----------
        config: ConfigFile
            a dictionary with values necessary to retrieve the data, such as
            the data source and preprocessing parameters

        Returns
        -------
        XYData
            data for evaluating the model. This can take different forms depending on the
            ML framework you use. For Tensorflow, it would look as x_test and y_test, while
            for PyTorch, it would look as a single DataLoader instance.

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_fit(self) -> NDArrays:
        """Fit the provided global model using a local dataset

        Parameters
        ----------

        Parameters for fitting the model. This might take different forms depending on the
        ML framework. For Tensorflow, it might look like this:
            model: keras.Sequential
            ModelTrainer: Class
            config: ConfigFile
            x_train: NDArrays
            y_train: Union[NDArray, NDArrays]

        Returns
        -------
        NDArrays
            new model weights as Numpy arrays

        """

        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_send(self) -> dict:
        """Final data processing before sending the results back, such as encryption.

        Returns
        -------
        results: dict
            FL results formatted as a dictionary. For example:
            results = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
            }

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def run_round(self, model_trainer, config: dict) -> dict:
        """Combines the rest of the functions to run a single Federated Learning round

        Parameters
        ----------
        model_trainer
            instance of a class based on BaseModelTrainer that implements
            .fit(), .evaluate(), .get_weights(), and .set_weights()

        config: dict
            a dictionary with all the necessary parameters for retrieving data and
            training the model

        Returns
        -------
        results: dict
            FL results formatted as a dictionary. For example:
            results = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
            }

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")
