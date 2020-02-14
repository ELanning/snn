r"""
Module for standard spiking neural network time course functions.
Functions follow the convention of taking a time parameter, followed by a list of spike times.
"""
import torch as t
from typing import List


def dirac_pulse(time: int, spike_history: List[t.Tensor]) -> t.Tensor:
    if time > len(spike_history):
        raise ValueError(
            "time must be less than or equal to spike_history length."
            f"Received time: {time}\n"
            f"spike_history length: {len(spike_history)}"
        )

    return spike_history[time].eq(1).float()
