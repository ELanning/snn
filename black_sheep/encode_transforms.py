"""
Transform wrappers for each encoding in the encode module.
Allows ease of use when using transforms.compose.
"""
import torch as t
from .encode import poisson_encode


class PoissonEncode:
    def __init__(self, spike_train_count: int):
        if spike_train_count < 1:
            raise ValueError("spike_train_count must be a positive integer.")

        self.spike_train_count = spike_train_count

    def __call__(self, data: t.Tensor):
        return poisson_encode(data, self.spike_train_count)

