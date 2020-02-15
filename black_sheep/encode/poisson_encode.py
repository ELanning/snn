import torch as t


def poisson_encode(data: t.Tensor, spike_train_count: int) -> t.Tensor:
    """
    Encodes the data into a tensor of Poisson spike trains.
    Formally, it is a function f: R -> {0, 1}ⁿ, where n is the spike_train_count for each data point.
    A 4x4 image would be encoded into a 16xN tensor: one spike train per pixel.
    Can be made deterministic by invoking torch.manual_seed before calling.

    Reference: https://arxiv.org/ftp/arxiv/papers/1604/1604.06751.pdf

    @param data: The data to encode.
    @param spike_train_count: The number of spikes in each Poisson spike train.
    @return: A float tensor of one Poisson spike train per element in the data.
    @raise ValueError: spike_train_count is not a positive integer.
    """
    if spike_train_count < 1:
        raise ValueError("spike_train_count must be a positive integer.")

    # Normalize the data, if necessary.
    if data.min() < 0:
        data += t.abs(data.min())
    if data.max() > 1.0:
        data /= data.max()

    total_spike_train_count = data.numel()
    # Typically a color pixel density, but could be anything.
    point_densities = (
        data.flatten()
        .repeat_interleave(spike_train_count)
        .view((total_spike_train_count, spike_train_count))
    )
    uniform_tensor = t.empty(total_spike_train_count, spike_train_count).uniform_(0, 1)
    result = point_densities.ge(uniform_tensor).float()
    return result
