import unittest
from black_sheep.network.layer import create_layer


class TestLayer(unittest.TestCase):
    def test_create_layer(self):
        input_count = 800
        output_count = 400
        [voltages, weights, _] = create_layer(input_count, output_count)

        # Validate dimensions.
        self.assertEqual(
            (input_count, output_count),
            weights.shape,
            f"weights dimension must be equal to {input_count}x{output_count}.\n"
            f"Received weights of shape {weights.shape}.",
        )

        self.assertEqual(
            (1, output_count),
            voltages.shape,
            f"voltages dimension must be equal to 1x{output_count}.\n"
            f"Received voltages of shape {voltages.shape}.",
        )

    def test_create_layer_bad_inputs(self):
        negative_parameter = -1
        empty_parameter = 0

        # Check input with bad parameters
        self.assertRaises(ValueError, create_layer, negative_parameter, 1)
        self.assertRaises(ValueError, create_layer, empty_parameter, 1)

        # Check output with bad parameter.
        self.assertRaises(ValueError, create_layer, 1, negative_parameter)
        self.assertRaises(ValueError, create_layer, 1, empty_parameter)
