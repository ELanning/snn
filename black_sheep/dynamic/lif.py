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
    # Apply the time course function to the spike history.
    # Common time course functions include the Dirac delta function.
    time_course_result = time_course_func(time, spike_history)

    # Edge case check: prevents inputs of 1x1 or 1 from being destroyed by squeeze.
    is_1x1_edge_case = time_course_result.shape == (1, 1)
    is_1_dimension_edge_case = (
        len(time_course_result.shape) == 1 and time_course_result.shape[0] == 1
    )
    if not is_1x1_edge_case and not is_1_dimension_edge_case:
        time_course_result.squeeze_()

    # Rescale the time course function result into a dimension that can be multiplied by the weights.
    # For example, a 24x24 image is rescaled and transformed to a 1x784 (24 * 24 = 784) spike train.
    # This 1x784 spike train is then multiplied by a 784x400 layer of neurons, which represents a fully connected
    # layer of 400 neurons receiving stimulus from 784 input neurons, to get the total voltage of each output neuron.
    input_dimension = time_course_result.shape[0]
    output_dimension = weights.shape[1]
    rescaled_time_course_result = time_course_result.repeat(output_dimension).view(
        (input_dimension, output_dimension)
    )
    current_sum = t.sum(weights * rescaled_time_course_result, dim=0)
    return current_sum


def calculate_lif_derivative(voltages: t.Tensor, input_current: t.Tensor) -> t.Tensor:
    return -1 * voltages + input_current
