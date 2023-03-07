from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime

from flox.common.typing import NDArrays


@dataclass
class TaskData:
    local_id: str = None
    funcx_uuid: str = None
    future: Future = None
    # model_weights: NDArrays = None
    broadcast_start_timestamp: datetime = None
    broadcast_finish_timestamp: datetime = None
    future_completed_timestamp: datetime = None
    actual_n_samples: int = None
    # epochs: int = None
    # batch_size: int = None
    model_accuracy: float = None
    model_loss: float = None
    task_runtime: float = None
    task_start_timestamp: datetime = None
    task_finish_timestamp: datetime = None
    training_runtime: float = None
    data_processing_runtime: float = None
    broadcasted_to_retrieved_runtime: float = None
    file_size: int = None
    physical_memory: int = None
    physical_cores: int = None
    logical_cores: int = None
    endpoint_platform_name: str = None


@dataclass
class EndpointData:
    id: str = None
    endpoint_custom_name: str = None
    tasks_per_endpoint: int = None
    desired_n_samples: int = None
    epochs: int = None
    batch_size: int = None
    path_directory: str = None
    latest_status: str = None
    task_ids: list = field(default_factory=list)
