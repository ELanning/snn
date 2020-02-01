import torch as t
import torchvision as tv
from black_sheep.encode_transforms import PoissonEncode
from typing import Callable

time_steps = 10

# Normalize and encode data into spike trains.
image_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    PoissonEncode(time_steps)
])

train_data_set = tv.datasets.MNIST("./", train=True, transform=image_transform, target_transform=None, download=True)

hidden_neuron_count = 80
hidden_neuron_voltages = t.zeros((1, hidden_neuron_count))  # 1x80
input_neuron_count = 784
hidden_neuron_weights = t.empty((hidden_neuron_count, input_neuron_count)).uniform_(0, 1)  # 80x784


def calculate_lif_current(time: int, weights: t.Tensor, time_course_func: Callable[[t.Tensor], t.Tensor], last_fired_times: t.Tensor) -> t.Tensor:
    last_fired_time_step = last_fired_times[time]
    time_course_input = last_fired_times.new_full(last_fired_time_step.shape, time + 1) - last_fired_time_step
    time_course_result = time_course_func(time_course_input)
    current_sum = t.sum(weights * time_course_result, dim=1)
    return current_sum


def calculate_lif_derivative(membrane_potential: t.Tensor, input_current: t.Tensor) -> t.Tensor:
    return -1 * membrane_potential + input_current


def dirac_pulse(x: t.Tensor):
    return x.eq(0).float()


last_fired_times = t.zeros((time_steps, hidden_neuron_count, input_neuron_count))
reset_voltage = t.zeros(hidden_neuron_voltages.shape)
# Simulate the spiking neural network for 10 time slices using the Euler method.
for spike_trains, label in train_data_set:  # spike_train size: 784x10
    # For conceptual understanding, each time step can be considered as 1 second.
    for i in range(time_steps):
        # Broadcast each spike train slice to each hidden neuron.
        spike_train_step = spike_trains[:, i]\
            .repeat(hidden_neuron_count)\
            .view((hidden_neuron_count, input_neuron_count))  # 80x784
        last_fired_times[i, :, :] = spike_train_step
        current = calculate_lif_current(i, hidden_neuron_weights, dirac_pulse, last_fired_times)
        derivative = calculate_lif_derivative(hidden_neuron_voltages, current)
        voltage_gain = hidden_neuron_voltages + derivative
        hidden_neuron_voltages += voltage_gain

        spike_threshold = 65
        hidden_neuron_voltages = hidden_neuron_voltages.where(hidden_neuron_voltages < spike_threshold, reset_voltage)

