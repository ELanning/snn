import torch as t
import torchvision as tv
from black_sheep.encode_transforms import PoissonEncode

time_steps = 10

# Normalize and encode data into spike trains.
image_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    PoissonEncode
])

train_data_set = tv.datasets.MNIST("./", train=True, transform=image_transform, target_transform=None, download=True)

hidden_neuron_count = 80
hidden_neuron_voltages = t.zeros((1, hidden_neuron_count))  # 1x80
input_neuron_count = 784
hidden_neuron_weights = t.empty((hidden_neuron_count, input_neuron_count)).uniform_(0, 1)  # 80x784

step_size = 0.01

for spike_trains, label in train_data_set:  # spike_train size: 784x10
    for i in range(time_steps):
        input_spike_trains = spike_trains[:, i]\
            .repeat(hidden_neuron_count)\
            .view((hidden_neuron_count, input_neuron_count))\
            .float()
        voltage_gain = t.sum(input_spike_trains * hidden_neuron_weights, dim=1)
        hidden_neuron_voltages += voltage_gain
