import torch as t

ThresholdTensor = t.Tensor


def threshold_init(
    input_size: int, output_size: int, threshold: float
) -> ThresholdTensor:
    threshold_tensor = t.empty(input_size, output_size).fill_(threshold)
    return threshold_tensor
