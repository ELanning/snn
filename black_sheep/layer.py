r"""
Module for creating and manipulating layers of a spiking neural network.
"""
import torch as t
import warnings
from typing import Callable, List, Union, Optional
from .weight import normal_weight_init


class Layer:
    def __init__(
        self, voltages: t.Tensor, weights: t.Tensor, spike_history: List[t.Tensor]
    ):
        self._voltages: t.Tensor = voltages
        self._weights: t.Tensor = weights
        self._spike_history: List[t.Tensor] = spike_history

    @property
    def voltages(self) -> t.Tensor:
        return self._voltages

    @voltages.setter
    def voltages(self, value: t.Tensor):
        if self._voltages.shape != value.shape:
            warnings.warn(
                f"Changed voltage shape from {self._voltages.shape} to {value.shape}."
            )
        self._voltages = value

    @property
    def weights(self) -> t.Tensor:
        return self._weights

    @weights.setter
    def weights(self, value: t.Tensor):
        if self._weights.shape != value.shape:
            warnings.warn(
                f"Changed weights shape from {self._weights.shape} to {value.shape}."
            )
        self._weights = value

    @property
    def spike_history(self) -> List[t.Tensor]:
        return self._spike_history

    @spike_history.setter
    def spike_history(self, value: t.Tensor):
        self._spike_history = value

    @property
    def output_size(self) -> int:
        output_dimension = 1
        return self._weights.shape[output_dimension]

    @property
    def input_size(self) -> int:
        input_dimension = 0
        return self._weights.shape[input_dimension]

    def __iter__(self):
        return iter([self.voltages, self.weights, self.spike_history])


def reset(layer: Layer):
    layer.voltages.fill_(0)
    layer.spike_history = []


def append_spike_history(
    spike_history: Union[Layer, List[t.Tensor]], spike_train: t.Tensor
) -> None:
    if isinstance(spike_history, Layer):
        if spike_history.voltages.shape != spike_train.shape:
            warnings.warn(
                f"voltage shape of {spike_history.voltages.shape} differed from spike_train shape of {spike_train.shape}."
            )
        spike_history.spike_history.append(spike_train)

    if isinstance(spike_history, list):
        spike_history.append(spike_train)


# Defaults may change in the future. Do not expect them to be static.
def default_weight_init(input_size: int, output_size: int) -> t.Tensor:
    return normal_weight_init(input_size, output_size, connection_count=4)


default_connection_init = None


def create_layer(
    input_size: int,
    output_size: int,
    weight_init: Callable[[int, int], t.Tensor] = default_weight_init,
    mask_init: Optional[Callable[[int, int], t.Tensor]] = default_connection_init,
) -> Layer:
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")

    voltages = t.zeros((1, output_size))
    spike_history = []

    # Initialize dense weight matrix.
    weights = weight_init(input_size, output_size)

    # Prune it using the result of mask_init.
    if mask_init is not None:
        mask = mask_init(input_size, output_size)
        weights = t.zeros_like(weights).masked_scatter_(mask, weights)

    return Layer(voltages, weights, spike_history)
