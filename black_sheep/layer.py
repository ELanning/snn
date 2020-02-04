import torch as t
from torch.distributions import Bernoulli
from typing import Callable, Tuple, List


def normal_weight_init(input_size: int, output_size: int) -> t.Tensor:
    """
    Useful for creating a tensor of connection weights.
    The resulting weight tensor will be dense and should be pruned with a mask.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @return: A torch.FloatTensor of standard normal distributed weights of size input_size x output_size.
    @raise ValueError: input_size or output_size is not a positive integer.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")

    return t.normal(mean=0, std=1.0, size=(input_size, output_size))


def random_subset_mask_init(input_size: int, output_size: int) -> t.Tensor:
    """
    Creates a mask for use with a weight tensor, such that each neuron will receive on average
    four connections, with a Binomial variance that converges to four.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @return: A pruned torch.FloatTensor of size input_size x output_size,
        with each neuron having on average four connections.
    @raise ValueError: input_size or output_size is not a positive integer.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")

    # May result in some neurons that have zero connections to the input.
    # In practice, this is usually not an issue.
    connection_probability = (
        4 / output_size
    )  # 4 is chosen to select roughly 4 connections per output neuron.

    # Default to fully connected if output_size is small.
    if connection_probability > 1:
        connection_probability = 1

    if connection_probability < 0:
        raise FloatingPointError("connection_probability underflowed.")

    bernoulli_distribution = Bernoulli(t.tensor([connection_probability]))
    return bernoulli_distribution.sample((input_size, output_size))


default_weight_init = normal_weight_init
default_connection_init = random_subset_mask_init


def create_layer(
    input_size: int,
    output_size: int,
    weight_init: Callable[[int, int], t.Tensor] = default_weight_init,
    mask_init: Callable[[int, int], t.Tensor] = default_connection_init,
) -> Tuple[t.Tensor, t.Tensor, List]:
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")

    voltages = t.zeros((1, output_size))
    spike_history = []

    # Initialize dense weight matrix.
    weights = weight_init(input_size, output_size)

    # Prune it using the result of mask_init.
    mask = mask_init(input_size, output_size)
    weights = weights.masked_scatter_(mask)

    return voltages, weights, spike_history
