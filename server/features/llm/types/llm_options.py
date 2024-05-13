from typing import TypedDict

from server.types import ComputeTypes, Devices


class LLMOptions(TypedDict):
    """
    Summary
    -------
    the parameters to initialise the LLM

    Attributes
    ----------
    model_path (str) : the path to the model
    device (Devices) : the device to use
    compute_type (ComputeTypes) : the compute type
    inter_threads (int) : the number of inter-threads
    max_queued_batches (int) : the number of max queued batches
    """
    model_path: str
    device: Devices
    compute_type: ComputeTypes
    inter_threads: int
    max_queued_batches: int
