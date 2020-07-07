r"""
Module for simulating the dynamics of a leaky integrate and fire spiking neural network.
"""
import torch as t
from typing import Callable, List


def calculate_lif_current(
    time: int,
    weights: t.Tensor,
    spike_history: List[t.Tensor],
    time_course_func: Callable[[int, List[t.Tensor]], t.Tensor],
) -> t.Tensor:
    time_course_result = time_course_func(time, spike_history)

    # Rescale the time course function result into a dimension that can be multiplied by the weights.
    input_dimension = weights.shape[1]
    output_dimension = time_course_result.shape[0]
    rescaled_time_course_result = time_course_result.repeat(input_dimension).view(
        (output_dimension, input_dimension)
    )
    current_sum = t.sum(weights * rescaled_time_course_result, dim=0)
    return current_sum


def calculate_lif_derivative(voltages: t.Tensor, input_current: t.Tensor) -> t.Tensor:
    return -1 * voltages + input_current
