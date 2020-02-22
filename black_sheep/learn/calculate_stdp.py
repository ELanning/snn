import torch as t
from typing import List, Tuple
from ..encode import rescale_spike_train


def calculate_stdp(
    weights: t.Tensor,
    input_spike_history: List[t.Tensor],
    output_spike_history: List[t.Tensor],
    window_size: int,
) -> t.Tensor:
    input_history_len = len(input_spike_history)
    output_history_len = len(output_spike_history)
    if input_history_len != output_history_len:
        raise ValueError(
            "input_spike_history length and output_spike_history length must be equal."
        )

    weight_updates = t.zeros_like(weights)
    # Calculations can be skipped if there is not enough spike history.
    if input_history_len <= 1 or output_history_len <= 1:
        return weight_updates

    last_output_spike = output_history_len * output_spike_history[-1]
    if last_output_spike.max() != 0:
        for index, input_spike in enumerate(reversed(input_spike_history)):
            # Latest input spike can be skipped.
            # This is because if both spike at the exact same time, the result is zero.
            if index == 0:
                continue

            # Add one to the window_size because the last input spike is skipped.
            if index == window_size + 1:
                break

            # Skip calculation if there are no spikes.
            if input_spike.max() == 0:
                continue

            time_multiplier = input_history_len - index
            weight_updates += positive_window_function(
                weights, time_multiplier * input_spike, last_output_spike
            )

    last_input_spike = input_history_len * input_spike_history[-1]
    if last_input_spike.max() != 0:
        for index, output_spike in enumerate(reversed(output_spike_history)):
            # Latest output spike can be skipped.
            # This is because if both spike at the exact same time, the result is zero.
            if index == 0:
                continue

            # Add one to the window_size because the last output spike is skipped.
            if index == window_size + 1:
                break

            # Skip calculation if there are no spikes.
            if output_spike.max() == 0:
                continue

            time_multiplier = output_history_len - index
            weight_updates += negative_window_function(
                weights, last_input_spike, time_multiplier * output_spike
            )

    return weight_updates


def positive_window_function(
    weights: t.Tensor, input_spike_train: t.Tensor, output_spike_train: t.Tensor,
) -> t.Tensor:
    difference, mask = get_difference(input_spike_train, output_spike_train)
    max_weights = t.empty(weights.shape).fill_(3)
    result = 0.1 * (max_weights - weights) * t.exp(-difference / 10)
    masked_result = t.zeros_like(result).masked_scatter(mask, result)
    return masked_result


def negative_window_function(
    weights: t.Tensor, input_spike_train: t.Tensor, output_spike_train: t.Tensor,
) -> t.Tensor:
    difference, mask = get_difference(input_spike_train, output_spike_train)
    result = 0.1 * weights * t.exp(difference / 10)
    masked_result = t.zeros_like(result).masked_scatter(mask, result)
    return masked_result


Difference = t.Tensor
Mask = t.Tensor


def get_difference(
    input_spike_train: t.Tensor, output_spike_train: t.Tensor
) -> Tuple[Difference, Mask]:
    # Format output and input spike trains so they are of equal dimension.
    output_spike_train_size = output_spike_train.shape[0]
    rescaled_input = rescale_spike_train(input_spike_train, output_spike_train_size)

    input_spike_train_size = input_spike_train.shape[0]
    rescaled_output = t.t(
        rescale_spike_train(output_spike_train, input_spike_train_size)
    )

    assert rescaled_output.shape == rescaled_input.shape

    # Only count elements where both the input and the output spiked.
    non_output_spikes = rescaled_output == 0
    non_input_spikes = rescaled_input == 0
    mask = (
        t.zeros_like(rescaled_output)
        .where(non_output_spikes | non_input_spikes, t.ones_like(rescaled_output),)
        .bool()
    )
    difference = rescaled_output - rescaled_input

    return difference, mask
