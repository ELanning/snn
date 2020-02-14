r"""
Module for simulating the dynamics of a leaky integrate and fire spiking neural network.
"""
import torch as t
from typing import Callable, List, Union
from .layer import Layer


def calculate_lif_current(
    time: int,
    weights: t.Tensor,
    spike_history: List[t.Tensor],
    time_course_func: Callable[[int, List[t.Tensor]], t.Tensor],
) -> t.Tensor:
    time_course_result = time_course_func(time, spike_history)
    current_sum = t.sum(weights * time_course_result, dim=0)
    return current_sum


def calculate_lif_derivative(
    membrane_potential: t.Tensor, input_current: t.Tensor
) -> t.Tensor:
    return -1 * membrane_potential + input_current


def calculate_spikes(layer: Layer, spike_threshold: float) -> t.Tensor:
    ones = t.ones_like(layer.voltages)
    zeros = t.zeros_like(layer.voltages)
    spikes = zeros.clone().where(layer.voltages < spike_threshold, ones)
    return spikes


def reset_where_spiked(layer: Layer, spike_threshold: Union[float, t.Tensor]) -> None:
    zeros = t.zeros_like(layer.voltages)
    layer.voltages = layer.voltages.where(layer.voltages < spike_threshold, zeros)
