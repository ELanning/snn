import torch as t
from torch.nn.init import sparse_
from torch.distributions import Bernoulli
from typing import Optional


def normal_weight_init(
    input_size: int,
    output_size: int,
    connection_count: Optional[int] = None,
    positive_ratio: Optional[float] = None,
) -> t.Tensor:
    """
    Creates a weight matrix using the standard normal distribution.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @param connection_count: The connection count per output neuron. Defaults to fully connected.
    @param positive_ratio: The approximate ratio of connections that will be positive.
    @return: A float tensor of standard normal distributed weights of size input_size x output_size.
    @raise ValueError:
        input_size is not a positive integer.
        output_size is not a positive integer.
        connection_count is not greater than or equal to one, or connection count is greater than output_size.
        positive_ratio is not None or between zero and one.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if connection_count is not None:
        if connection_count < 1:
            raise ValueError(
                "connection_count must be None, or must be greater than or equal to one."
            )
        if connection_count > output_size:
            raise ValueError(
                "connection_count must be None, or must not be greater than output_size."
            )
    if positive_ratio is not None:
        if positive_ratio > 1 or 0 > positive_ratio:
            raise ValueError(
                "positive_ratio must be None, or must be between zero and one."
            )

    result = t.empty((input_size, output_size))

    sparsity = 0.0
    if connection_count is not None:
        sparsity = 1 - (connection_count / output_size)

    sparse_(result, sparsity=sparsity, std=1.0)

    # TODO: test this.
    if positive_ratio is not None:
        bernoulli_distribution = Bernoulli(t.tensor([positive_ratio]))
        mask = bernoulli_distribution.sample((input_size, output_size)).squeeze().bool()
        result.abs_()
        result = result.where(mask, -result)

    return result
