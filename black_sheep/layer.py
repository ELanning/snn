import torch as t
from torch.distributions import Bernoulli
from typing import Callable, List


class Layer:
    def __init__(
        self, voltages: t.Tensor, weights: t.Tensor, spike_history: List[t.Tensor]
    ):
        self.voltages = voltages
        self.weights = weights
        self.spike_history = spike_history

    def __iter__(self):
        return iter([self.voltages, self.weights, self.spike_history])


def get_output_size(layer: Layer):
    output_dimension = 1
    return layer.weights.shape[output_dimension]


def normal_weight_init(input_size: int, output_size: int) -> t.Tensor:
    """
    Useful for creating a tensor of connection weights.
    The resulting weight tensor will be dense and should be pruned with a mask.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @return: A float tensor of standard normal distributed weights of size input_size x output_size.
    @raise ValueError: input_size or output_size is not a positive integer.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")

    return t.normal(mean=0, std=1.0, size=(input_size, output_size))


def random_subset_mask_init(
    input_size: int, output_size: int, connection_count: int = 4
) -> t.Tensor:
    """
    Creates a mask for use with a weight tensor, such that each neuron will receive on average
    connection_count connections, with a Binomial variance that converges to connection_count.
    If input_size >> output_size, expect approximately input_size * 0.001 connections.
    If output_size >> input_size, expect full connectivity.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @param connection_count: The average number of pre-synaptic connections per output neuron.
    @return: A bool tensor of size input_size x output_size,
        with each neuron having on average connection_count connections.
        Imbalanced sizes will result in different connection schemes.
    @raise ValueError: input_size, output_size, or connection_count is not a positive integer.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if connection_count < 1:
        raise ValueError("connection_count must be a positive integer.")

    # May result in some neurons that have zero connections to the input.
    # In practice, this is usually not an issue.
    # Calculate probability such that each output neuron will have on average connection_count connections.
    connection_probability = (connection_count * output_size) / (
        input_size * output_size
    )

    # Defaults if parameters are largely imbalanced.
    if connection_probability > 1:
        connection_probability = 1
    if connection_probability == 0:
        connection_probability = 0.001

    bernoulli_distribution = Bernoulli(t.tensor([connection_probability]))
    # squeeze with dim=2 removes the last dimension that sample adds.
    return (
        bernoulli_distribution.sample((input_size, output_size)).squeeze(dim=2).bool()
    )


# Defaults may change in the future. Do not expect them to be static.
default_weight_init = normal_weight_init
default_connection_init = random_subset_mask_init


def create_layer(
    input_size: int,
    output_size: int,
    weight_init: Callable[[int, int], t.Tensor] = default_weight_init,
    mask_init: Callable[[int, int], t.Tensor] = default_connection_init,
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
    mask = mask_init(input_size, output_size)
    weights = t.zeros_like(weights).masked_scatter_(mask, weights)

    return Layer(voltages, weights, spike_history)
