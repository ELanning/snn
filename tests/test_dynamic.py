import unittest
import torch as t
from black_sheep.dynamic import (
    calculate_lif_derivative,
    calculate_lif_current,
)
from black_sheep.time_course import dirac_pulse
from black_sheep.network.layer import create_layer


class TestDynamic(unittest.TestCase):
    def test_lif_constant_current(self):
        # Validate Leaky Integrate and Fire dynamics on a single neuron,
        # using the Euler method with a constant input current.
        layer = create_layer(1, 1)
        arbitrary_weight = 0.2125
        layer.weights = t.tensor([[arbitrary_weight]])
        constant_input_spike_history = []

        for time_step in range(4):
            constant_input_spike_history.append(
                t.tensor([1])
            )  # One indicates the neuron spiked.
            current = calculate_lif_current(
                time_step, layer.weights, constant_input_spike_history, dirac_pulse
            )
            derivative = calculate_lif_derivative(layer.voltages, current)
            layer.voltages = layer.voltages + derivative

            neuron_voltage = layer.voltages[0][0]
            expected_constant_voltage = 0.2125
            self.assertEqual(
                expected_constant_voltage,
                neuron_voltage,
                f"neuron_voltage must be equal to {expected_constant_voltage}.\n"
                f"Received a voltage of {neuron_voltage}",
            )
