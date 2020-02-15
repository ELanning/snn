import torch as t
from torch.nn.init import sparse_


def normal_weight_init(
    input_size: int, output_size: int, connection_count: int = 4
) -> t.Tensor:
    """
    Creates a weight matrix using the standard normal distribution.
    Each neuron will have connection_count connections to the input.

    @param input_size: The input layer size.
    @param output_size: The output layer size.
    @param connection_count: The connection count per output neuron.
    @return: A float tensor of standard normal distributed weights of size input_size x output_size.
    @raise ValueError: input_size or output_size is not a positive integer.
        connection_count is not greater than or equal to one.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer.")
    if output_size < 1:
        raise ValueError("output_size must be a positive integer.")
    if connection_count < 1:
        raise ValueError("connection_count must be greater than or equal to one.")

    result = t.empty((input_size, output_size))

    sparsity = 1 - (connection_count / output_size)
    if sparsity > 1.0:
        sparsity = 1.0

    return sparse_(result, sparsity=sparsity, std=1.0)
