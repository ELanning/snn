import torch as t
from typing import List
from ..encode import rescale_spike_train


def calculate_stdp(
    weights: t.Tensor,
    input_spike_history: List[t.Tensor],
    output_spike_history: List[t.Tensor],
    window_size: int,
) -> t.Tensor:
    weight_updates = t.zeros_like(weights)

    last_output_spike = output_spike_history[-1]
    for i in range(window_size):
        input_spike = input_spike_history[-(i + 1)]
        weight_updates += window_func(weights, input_spike, last_output_spike)

    last_input_spike = input_spike_history[-1]
    for i in range(window_size):
        output_spike = output_spike_history[-(i + 1)]
        weight_updates += window_func(weights, last_input_spike, output_spike)

    return weight_updates


def window_func(
    weights: t.Tensor, input_spike_train: t.Tensor, output_spike_train: t.Tensor
):
    # TODO: Validate input_spike_train and output_spike_train.
    output_spike_train_size = output_spike_train.shape[1]
    rescaled_input = rescale_spike_train(input_spike_train, output_spike_train_size)

    input_spike_train_size = input_spike_train.shape[1]
    transposed_output = t.t(output_spike_train)
    rescaled_output = rescale_spike_train(transposed_output, input_spike_train_size)

    assert rescaled_output.shape == weights.shape
    assert rescaled_input.shape == weights.shape

    mask = (
        t.zeros_like(rescaled_output)
        .where(
            rescaled_output == 0 or rescaled_input == 0, t.ones_like(rescaled_output)
        )
        .bool()
    )
    difference = rescaled_output - rescaled_input
    result = t.exp(difference)
    masked_result = t.zeros_like(result).masked_scatter(mask, result)

    return masked_result
