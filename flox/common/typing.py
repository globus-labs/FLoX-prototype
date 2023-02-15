"""FLoX Type Definitions"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    loss: float
    metrics: Dict[str, Scalar]


@dataclass
class FuturesList:
    """List of futures. If using FuncXExecutor, this would most likely be a list of futures
    funcX returns after you submit functions to endpoints"""

    tasks: List[Any]


@dataclass
class ResultsList:
    """FL results extracted from FuturesList, formatted as a dictionary. For example:
    ResultsList = {
        "model_weights": model_weights,
        "samples_count": samples_count,
        "bias_weights": fractions,
    }
    """

    results: Dict[str, Scalar]


@dataclass
class UpdateIns:
    """List of inputs for on_model_update"""

    weights: NDArrays


@dataclass
class ConfigFile:
    """Dictionary with the necessary inputs for functions like on_data_retrieve"""

    config: Dict[str, Scalar]


@dataclass
class XYData:
    """x_train/test and y_train/test data for model training or testing"""

    x_data: NDArrays
    y_data: Union[NDArray, NDArrays]


@dataclass
class FitIns:
    """Parameters for fitting the model"""

    model: Any
    ModelTrainer: Any
    config: ConfigFile
    x_train: NDArrays
    y_train: Union[NDArray, NDArrays]
