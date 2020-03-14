import torch as t
from typing import Union
from black_sheep.network.layer import Layer


def calculate_spikes(
    voltages: t.Tensor, spike_threshold: Union[float, t.Tensor]
) -> t.Tensor:
    ones = t.ones_like(voltages)
    zeros = t.zeros_like(voltages)
    spikes = zeros.where(voltages < spike_threshold, ones)
    return spikes


def reset_where_spiked_(layer: Layer, spike_threshold: Union[float, t.Tensor]) -> None:
    zeros = t.zeros_like(layer.voltages)
    layer.voltages = layer.voltages.where(layer.voltages < spike_threshold, zeros)
