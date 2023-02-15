"""Abstract Base Class for FLoX Controllers"""
from flox.common import EvaluateRes, FuturesList, NDArrays, ResultsList


class FloxControllerLogic:
    """Abstract base class for FLoX Controller logic"""

    def on_model_init(self) -> None:
        """Does initial Controller setup before running the main Federated Learning loop"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_broadcast(self) -> FuturesList:
        """Sends the model and config to endpoints for FL training.

        Returns
        -------
        FuturesList
            A list of tasks/futures with results of the FL training returned from the endpoints.
            If using FuncXExecutor, this would most likely be a list of futures
            funcX returns after you submit functions to endpoints.

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_receive(self, ins: FuturesList) -> ResultsList:
        """Processes returned tasks from on_model_broadcast.

        Parameters
        ----------
        FuturesList
            A list of tasks/futures with results of the FL training returned from the endpoints.
            If using FuncXExecutor, this would most likely be a list of futures
            funcX returns after you submit functions to endpoints.

        Returns
        -------
        ResultsList
            FL results extracted from FuturesList, formatted as a dictionary. For example:
            ResultsRes = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
            }
        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_aggregate(self, ins: ResultsList) -> NDArrays:
        """Aggregates weights.

        Parameters
        ----------
        ResultsList
            FL results extracted from FuturesList, formatted as a dictionary. For example:
            ResultsRes = {
                "model_weights": model_weights,
                "samples_count": samples_count,
                "bias_weights": fractions,
            }

        Returns
        -------
        NDArrays
            ML model weights for a single model in the form of Numpy Arrays.

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_update(self, weights: NDArrays) -> None:
        """Updates the model's weights with new weights

        Parameters
        ----------
        weights: NDArrays
            ML model weights for a single model in the form of Numpy Arrays.

        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_evaluate(self, test_data, model=None) -> EvaluateRes:
        """Evaluates the given model using test_data

        Parameters
        ----------
        test_data
            data for evaluating the model. This can take different forms depending on the
            ML framework you use. For Tensorflow, it would look as x_test and y_test, while
            for PyTorch, it would look as a single DataLoader instance.

        model
            The machine learning model for evaluation. Whether you pass it as a parameter
            would depend on the ML framework. With Tensorflow, we cannot save and transfer
            the model as a class attribute, while for PyTorch it was possible so we do not need
            to pass it as a paramete.

        Returns
        -------
        EvaluateRes
            A class consisting of two attributes showing the loss and metrics of the evaluated
            model on the dataset:
                loss: float
                metrics: Dict[str, Scalar]
        """
        raise NotImplementedError("Abstract class method. Cannot be called directly.")
