import torch as t
from torch.distributions.normal import Normal
import torchvision as tv
from typing import List, Tuple
from black_sheep.encode import poisson_encode
from black_sheep.weight import xavier_normal_weight_init
from black_sheep.network.layer import (
    Layer,
    create_layer,
    reset_,
    append_spike_history,
)
from black_sheep.dynamic import (
    calculate_lif_derivative,
    calculate_lif_current,
    calculate_spikes,
    reset_where_spiked_,
)
from black_sheep.constant.diehl_cook import excitatory_time_constant_ms
from black_sheep.time_course import dirac_pulse

image_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
train_data_set = tv.datasets.MNIST(
    "./", train=False, transform=image_transform, target_transform=None, download=True
)


def positive_xavier_normal_weight_init(input_size, output_size):
    return xavier_normal_weight_init(input_size, output_size, positive_ratio=1.0)


time_step_count = 10
input_dimension = 784

candidates: List[Tuple[Layer, Layer, int]] = []
candidate_count = 10

for i in range(candidate_count):
    hidden_layer_neuron_count = 400
    hidden_layer = create_layer(input_dimension, hidden_layer_neuron_count)
    hidden_layer.weights = positive_xavier_normal_weight_init(
        hidden_layer.input_size, hidden_layer.output_size
    )

    # One neuron per class label (0-9).
    output_layer_neuron_count = 10
    output_layer = create_layer(hidden_layer.output_size, output_layer_neuron_count,)
    output_layer.weights = positive_xavier_normal_weight_init(
        output_layer.input_size, output_layer.output_size
    )

    candidates.append((hidden_layer, output_layer, i))

# candidates[0][0].weights = t.load("./hidden_weights.pt")
# candidates[0][1].weights = t.load("./output_weights.pt")


def train():
    generation_count = 100
    best = None
    for generation in range(generation_count):
        fitness_scores = []

        for h_layer, o_layer, index in candidates:
            fitness_score = 0

            for image_vector, label in train_data_set:
                # Reset layers.
                input_spike_history: List[t.Tensor] = []
                reset_(h_layer)
                reset_(o_layer)

                # Simulate the Euler method for time_step_count steps.
                # For conceptual understanding, each time step can be considered as 1 second.
                for spikes, time_step in poisson_encode(image_vector, time_step_count):
                    append_spike_history(input_spike_history, spikes)

                    # Calculate hidden neuron dynamic.
                    hidden_spike_threshold = 0.1
                    hidden_layer_spikes = run_lif_simulation_step(
                        h_layer, input_spike_history, time_step, hidden_spike_threshold,
                    )
                    append_spike_history(h_layer, hidden_layer_spikes.squeeze())

                    # Calculate output neuron dynamic.
                    output_spike_threshold = (
                        1000  # Arbitrarily high number so the output never spikes.
                    )
                    run_lif_simulation_step(
                        o_layer,
                        h_layer.spike_history,
                        time_step,
                        output_spike_threshold,
                    )
                _, rankings = o_layer.voltages.sort()
                fitness_score += rankings[0][label].item()

            fitness_score /= len(train_data_set)
            fitness_scores.append(fitness_score)

        # Mutate candidates.
        top_candidate_index = fitness_scores.index(max(fitness_scores))
        print(fitness_scores[top_candidate_index])
        normal_distribution = Normal(t.tensor([0.0]), t.tensor([1.0]))
        best = candidates[top_candidate_index]

        for h_layer, o_layer, index in candidates:
            if index == top_candidate_index:
                continue

            h_layer.weights = best[0].weights + 0.1 * normal_distribution.sample(
                h_layer.weights.shape
            ).squeeze(dim=2)
            o_layer.weights = best[1].weights + 0.3 * normal_distribution.sample(
                o_layer.weights.shape
            ).squeeze(dim=2)

    t.save(best[0].weights, f"hidden_weights.pt")
    t.save(best[1].weights, f"output_weights.pt")


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
    layer.voltages = layer.voltages + (derivative / excitatory_time_constant_ms)

    spikes = calculate_spikes(layer.voltages, spike_threshold)
    reset_where_spiked_(layer, spike_threshold)

    return spikes


if __name__ == "__main__":
    train()
