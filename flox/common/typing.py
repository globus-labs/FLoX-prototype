"""FLoX Type Definitions"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

Scalar = Union[bool, bytes, float, int, str]

Metrics = Dict[str, Scalar]


@dataclass
class XYData:
    """x_train/test and y_train/test data for model training or testing"""

    x_data: NDArrays
    y_data: Union[NDArray, NDArrays]
