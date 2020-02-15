"""
Transform wrapper for poisson_encode
Allows ease of use when using transforms.compose.
"""
import torch as t
from black_sheep.encode.poisson_encode import poisson_encode


class PoissonEncode:
    def __init__(self, spike_train_count: int):
        if spike_train_count < 1:
            raise ValueError("spike_train_count must be a positive integer.")

        self.spike_train_count = spike_train_count

    def __call__(self, data: t.Tensor):
        return poisson_encode(data, self.spike_train_count)
