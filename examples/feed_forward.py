import torch as t
import torchvision as tv
from black_sheep.encode import rescale_spike_train, PoissonEncode
from black_sheep.weight import normal_weight_init
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


def fully_connected_weight_init(input_size, output_size):
    return normal_weight_init(input_size, output_size, connection_count=output_size)


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
output_layer = create_layer(
    hidden_layer.output_size, output_layer_neuron_count, fully_connected_weight_init
)


def train():
    for spike_trains, label in train_data_set:
        # Reset layers.
        reset(hidden_layer)
        reset(output_layer)

        # Simulate the Euler method for time_step_count steps.
        # For conceptual understanding, each time step can be considered as 1 second.
        for time_step in range(time_step_count):
            # Calculate hidden neuron dynamics.
            hidden_spike_threshold = 3
            hidden_layer_spikes = run_lif_simulation_step(
                hidden_layer,
                spike_trains[:, time_step],
                time_step,
                hidden_spike_threshold,
            )

            # Calculate output neuron dynamics.
            output_spike_threshold = 4
            output_layer_spikes = run_lif_simulation_step(
                output_layer, hidden_layer_spikes[0], time_step, output_spike_threshold
            )


# Mutates the passed in layers voltages.
def run_lif_simulation_step(
    layer: Layer, spike_train: t.Tensor, time_step: int, spike_threshold: float
) -> t.Tensor:
    rescaled_spike_train = rescale_spike_train(spike_train, layer.output_size)
    append_spike_history(layer.spike_history, rescaled_spike_train)

    current = calculate_lif_current(
        time_step, layer.weights, layer.spike_history, dirac_pulse
    )
    derivative = calculate_lif_derivative(layer.voltages, current)
    voltage_gain = layer.voltages + derivative
    layer.voltages += voltage_gain

    spikes = calculate_spikes(layer, spike_threshold)
    reset_where_spiked(layer, spike_threshold)

    return spikes


if __name__ == "__main__":
    train()
