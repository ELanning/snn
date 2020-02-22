import torch as t
from torch.nn.init import xavier_uniform_
from torch.distributions import Bernoulli
from typing import Optional


def xavier_uniform_weight_init(
    input_size: int, output_size: int, positive_percent: Optional[float] = None,
) -> t.Tensor:
    """
    Creates a weight matrix using the Xavier Uniform method.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @param positive_percent: The rough percentage of connections that will be positive.
    @return: A float tensor of Xavier Uniform distributed weights of size input_size x output_size.
    @raise ValueError:
        input_size is not a positive integer.
        output_size is not a positive integer.
        connection_count is not greater than or equal to one, or connection count is greater than output_size.
        positive_percent is not None or between zero and one.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if positive_percent is not None:
        if positive_percent > 1 or 0 > positive_percent:
            raise ValueError(
                "positive_percent must be None, or must be between zero and one."
            )

    result = t.empty((input_size, output_size))
    xavier_uniform_(result)

    # TODO: test this.
    if positive_percent is not None:
        bernoulli_distribution = Bernoulli(t.tensor([positive_percent]))
        mask = bernoulli_distribution.sample((input_size, output_size)).squeeze().bool()
        result.abs_()
        result = result.where(mask, -result)

    return result
