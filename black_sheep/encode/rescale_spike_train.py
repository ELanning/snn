import torch as t


def rescale_spike_train(spike_train: t.Tensor, output_size: int) -> t.Tensor:
    """
    Rescale spike_train to output_size.
    Useful for broadcasting the spike train to n output neurons.

    @param spike_train: The spike train to rescale.
    @param output_size: The number of output neurons to rescale to.
    @return: The spike train rescaled to dimension size of spike train length x output size.
    """
    spike_train_length = spike_train.numel()
    return spike_train.repeat(output_size).view((spike_train_length, output_size))


def rescale_spike_train_(spike_train: t.Tensor, output_size: int) -> t.Tensor:
    """
    Rescale spike_train to output_size.
    Useful for broadcasting the spike train to n output neurons.

    @param spike_train: The spike train to rescale.
    @param output_size: The number of output neurons to rescale to.
    @return: The spike train rescaled to dimension size of spike train length x output size.
    """
    spike_train_length = spike_train.numel()
    return spike_train.expand(output_size).view((spike_train_length, output_size))
