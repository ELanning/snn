import torch as t
import torchvision as tv
from black_sheep.encode_transforms import PoissonEncode
from black_sheep.layer import create_layer
from typing import List, Callable

time_step_count = 10

# Normalize and encode data into spike trains.
image_transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(), PoissonEncode(time_step_count)]
)

train_data_set = tv.datasets.MNIST(
    "./", train=True, transform=image_transform, target_transform=None, download=True
)

[spike_train_size, _] = train_data_set[0][0].shape

layer1_neuron_count = 64
[layer1_voltages, layer1_weights, layer1_spike_history] = create_layer(
    spike_train_size, layer1_neuron_count
)


def calculate_lif_current(
    time: int,
    weights: t.Tensor,
    spike_history: List[t.Tensor],
    time_course_func: Callable[[int, List[t.Tensor]], t.Tensor],
) -> t.Tensor:
    time_course_result = time_course_func(time, spike_history)
    current_sum = t.sum(weights * time_course_result, dim=0)
    return current_sum


def calculate_lif_derivative(
    membrane_potential: t.Tensor, input_current: t.Tensor
) -> t.Tensor:
    return -1 * membrane_potential + input_current


def dirac_pulse(time: int, spike_history: List[t.Tensor]) -> t.Tensor:
    if time > len(spike_history):
        raise ValueError(
            f"time must be less than or equal to spike_history length."
            f"Received time: {time}\n spike_history length: {len(spike_history)}"
        )

    return spike_history[time].eq(1).float()


def rescale_spike_train(spike_train: t.Tensor, output_size: t.Tensor) -> t.Tensor:
    # Rescale spike_train to output_size.
    # Useful for broadcasting the spike train to n output neurons.
    spike_train_length = spike_train.numel()
    return spike_train.repeat(output_size).view((spike_train_length, output_size))


def update_spike_history(spike_history: List[t.Tensor], spike_train: t.Tensor) -> None:
    spike_history.append(spike_train)


reset_voltage = t.zeros_like(layer1_voltages)
for spike_trains, label in train_data_set:
    # Reset layers.
    # layer1_voltages = reset_voltage.clone() <--- this line is causing errors?
    layer1_spike_history = []

    # Simulate the Euler method for time_step_count
    # For conceptual understanding, each time step can be considered as 1 second.
    for i in range(time_step_count):
        spike_train_step = rescale_spike_train(spike_trains[:, i], layer1_neuron_count)
        update_spike_history(layer1_spike_history, spike_train_step)

        current = calculate_lif_current(
            i, layer1_weights, layer1_spike_history, dirac_pulse
        )
        derivative = calculate_lif_derivative(layer1_voltages, current)
        voltage_gain = layer1_voltages + derivative
        layer1_voltages += voltage_gain

        spike_threshold = 65
        layer1_voltages = layer1_voltages.where(
            layer1_voltages < spike_threshold, reset_voltage
        )
