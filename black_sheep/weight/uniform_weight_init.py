import torch as t
from torch.nn.init import uniform_
from torch.distributions import Bernoulli
from typing import Optional


def uniform_weight_init(
    input_size: int,
    output_size: int,
    min_bounds: float = 0.0,
    max_bounds: float = 1.0,
    positive_percent: Optional[float] = None,
) -> t.Tensor:
    """
    Creates a weight matrix using the uniform distribution.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @param min_bounds: The minimum of the uniform distribution. Defaults to 0.0
    @param max_bounds: The maximum of the uniform distribution. Defaults to 1.0
    @param positive_percent: The rough percentage of connections that will be positive.
    @return: A float tensor of uniform distributed weights of size input_size x output_size.
    @raise ValueError:
        input_size is not a positive integer.
        output_size is not a positive integer.
        min_bounds is greater than max_bounds.
        positive_percent is not None or between zero and one.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if min_bounds > max_bounds:
        raise ValueError("min_bounds must not be greater than max_bounds.")
    if positive_percent is not None:
        if positive_percent > 1 or 0 > positive_percent:
            raise ValueError(
                "positive_percent must be None, or must be between zero and one."
            )

    result = t.empty((input_size, output_size))
    uniform_(result, a=min_bounds, b=max_bounds)

    # TODO: test this.
    if positive_percent is not None:
        bernoulli_distribution = Bernoulli(t.tensor([positive_percent]))
        mask = bernoulli_distribution.sample((input_size, output_size)).squeeze().bool()
        result.abs_()
        result = result.where(mask, -result)

    return result
