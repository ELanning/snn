import torch as t
import torchvision as tv
from typing import List
from black_sheep.encode import PoissonEncode
from black_sheep.weight import xavier_normal_weight_init
from black_sheep.layer import (
    Layer,
    create_layer,
    reset,
    append_spike_history,
)
from black_sheep.lif import (
    calculate_lif_derivative,
    calculate_lif_current,
    calculate_spikes,
    reset_where_spiked,
)
from black_sheep.time_course import dirac_pulse
from black_sheep.learn import calculate_stdp

time_step_count = 10

# Normalize and encode data into spike trains.
image_transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(), PoissonEncode(time_step_count)]
)

train_data_set = tv.datasets.MNIST(
    "./", train=True, transform=image_transform, target_transform=None, download=True
)

[spike_train_size, _] = train_data_set[0][0].shape


def mostly_positive_xavier_normal(input_size, output_size):
    return xavier_normal_weight_init(input_size, output_size, positive_percent=0.9)


hidden_layer_neuron_count = 64
hidden_layer = create_layer(
    spike_train_size, hidden_layer_neuron_count, mostly_positive_xavier_normal
)

# One output neuron per class label (numbers zero to nine).
output_layer_neuron_count = 10
output_layer = create_layer(
    hidden_layer.output_size, output_layer_neuron_count, mostly_positive_xavier_normal
)


def train():
    for spike_trains, label in train_data_set:
        # Reset layers.
        input_spike_history: List[t.Tensor] = []
        hidden_weight_update = t.zeros_like(hidden_layer.weights)
        output_weight_update = t.zeros_like(output_layer.weights)
        reset(hidden_layer)
        reset(output_layer)

        # Simulate the Euler method for time_step_count steps.
        # For conceptual understanding, each time step can be considered as 1 second.
        for time_step in range(time_step_count):
            append_spike_history(input_spike_history, spike_trains[:, time_step])

            # Calculate hidden neuron dynamics.
            hidden_spike_threshold = 15
            hidden_layer_spikes = run_lif_simulation_step(
                hidden_layer, input_spike_history, time_step, hidden_spike_threshold,
            )
            append_spike_history(hidden_layer, hidden_layer_spikes.squeeze())

            # Calculate output neuron dynamics.
            output_spike_threshold = 4
            output_layer_spikes = run_lif_simulation_step(
                output_layer,
                hidden_layer.spike_history,
                time_step,
                output_spike_threshold,
            )
            append_spike_history(output_layer, output_layer_spikes.squeeze())

            # Update weights with the standard STDP algorithm.
            window_size = 3
            hidden_weight_update += calculate_stdp(
                hidden_layer.weights,
                input_spike_history,
                hidden_layer.spike_history,
                window_size,
            )
            hidden_layer.weights += hidden_weight_update
            output_weight_update += calculate_stdp(
                output_layer.weights,
                hidden_layer.spike_history,
                output_layer.spike_history,
                window_size,
            )
            output_layer.weights += output_weight_update


# Mutates the passed in layers voltages.
def run_lif_simulation_step(
    layer: Layer,
    previous_layer_spike_history: List[t.Tensor],
    time_step: int,
    spike_threshold: float,
) -> t.Tensor:
    current = calculate_lif_current(
        time_step, layer.weights, previous_layer_spike_history, dirac_pulse
    )
    derivative = calculate_lif_derivative(layer.voltages, current)
    voltage_gain = layer.voltages + derivative
    layer.voltages += voltage_gain

    spikes = calculate_spikes(layer, spike_threshold)
    reset_where_spiked(layer, spike_threshold)

    return spikes


if __name__ == "__main__":
    train()
