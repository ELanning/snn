import torch as t
import torchvision as tv
from black_sheep.encode_transforms import PoissonEncode
from black_sheep.layer import create_layer, get_output_size
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

hidden_layer_neuron_count = 64
hidden_layer = create_layer(spike_train_size, hidden_layer_neuron_count)

# One output neuron per class label (numbers zero to nine).
output_layer_neuron_count = 10
output_layer = create_layer(get_output_size(hidden_layer), output_layer_neuron_count)


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
            "time must be less than or equal to spike_history length."
            f"Received time: {time}\n"
            f"spike_history length: {len(spike_history)}"
        )

    return spike_history[time].eq(1).float()


def rescale_spike_train(spike_train: t.Tensor, output_size: int) -> t.Tensor:
    # Rescale spike_train to output_size.
    # Useful for broadcasting the spike train to n output neurons.
    spike_train_length = spike_train.numel()
    return spike_train.repeat(output_size).view((spike_train_length, output_size))


def update_spike_history(spike_history: List[t.Tensor], spike_train: t.Tensor) -> None:
    # TODO: change spike_history to a layer so can check for dimensions, or make it Layer or List.
    # TODO: Rename to append_spike_history?
    spike_history.append(spike_train)


zeros = t.zeros_like(hidden_layer.voltages)
ones = t.ones_like(hidden_layer.voltages)
for spike_trains, label in train_data_set:
    # Reset layers.
    hidden_layer.voltages = zeros.clone()
    hidden_layer.spike_history = []
    output_layer.voltages = t.zeros_like(output_layer.voltages).clone()
    output_layer.spike_history = []

    # Simulate the Euler method for time_step_count steps.
    # For conceptual understanding, each time step can be considered as 1 second.
    for i in range(time_step_count):
        # Calculate hidden neuron dynamics.
        input_spike_train_step = rescale_spike_train(
            spike_trains[:, i], hidden_layer_neuron_count
        )
        update_spike_history(hidden_layer.spike_history, input_spike_train_step)

        current = calculate_lif_current(
            i, hidden_layer.weights, hidden_layer.spike_history, dirac_pulse
        )
        derivative = calculate_lif_derivative(hidden_layer.voltages, current)
        voltage_gain = hidden_layer.voltages + derivative
        hidden_layer.voltages += voltage_gain

        spike_threshold = 3
        spikes = zeros.clone().where(hidden_layer.voltages < spike_threshold, ones)
        hidden_layer.voltages = hidden_layer.voltages.where(
            hidden_layer.voltages < spike_threshold, zeros
        )

        # Calculate output neuron dynamics.
        output_spike_train_step = rescale_spike_train(
            spikes[0], output_layer_neuron_count
        )
        update_spike_history(output_layer.spike_history, output_spike_train_step)

        current = calculate_lif_current(
            i, output_layer.weights, output_layer.spike_history, dirac_pulse
        )
        derivative = calculate_lif_derivative(output_layer.voltages, current)
        voltage_gain = output_layer.voltages + derivative
        output_layer.voltages += voltage_gain

        spike_threshold = 1.5
        spikes = t.zeros_like(output_layer.voltages).where(
            output_layer.voltages < spike_threshold, t.ones_like(output_layer.voltages)
        )
        output_layer.voltages = output_layer.voltages.where(
            output_layer.voltages < spike_threshold, t.zeros_like(output_layer.voltages)
        )
