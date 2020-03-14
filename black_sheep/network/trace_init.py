import torch as t

Trace = t.Tensor


def trace_init(input_size, output_size) -> Trace:
    trace = t.zeros(input_size, output_size)
    return trace
