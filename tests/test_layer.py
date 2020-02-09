import unittest
import torch as t
from black_sheep.layer import create_layer, normal_weight_init, random_subset_mask_init

arbitrary_number = 333
t.manual_seed(arbitrary_number)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False


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

    def test_normal_weight_init(self):
        input_count = 800
        output_count = 400
        normal_weights = normal_weight_init(input_count, output_count)

        # Validate dimensions.
        self.assertEqual(
            (input_count, output_count),
            normal_weights.shape,
            f"normal_weights dimension must be equal to {input_count}x{output_count}.\n"
            f"Received normal_weights of shape {normal_weights.shape}.",
        )

        # Validate weights are standard normal distributed.
        mean = normal_weights.mean()
        self.assertTrue(
            abs(mean < 0.01),
            "mean of normal_weights must be approximately equal to zero.\n"
            f"Received mean of {mean}",
        )

        variance = normal_weights.var()
        self.assertTrue(
            abs(variance - 1) < 0.01,
            "variance of normal_weights must be approximately equal to one.\n"
            f"Received variance of {variance}",
        )

    def test_random_subset_mask_init(self):
        input_count = 8
        output_count = 600
        random_subset_mask = random_subset_mask_init(input_count, output_count)

        # Validate dimensions.
        self.assertEqual(
            (input_count, output_count),
            random_subset_mask.shape,
            f"random_subset_mask dimension must be equal to {input_count}x{output_count}.\n"
            f"Received random_subset_mask of shape {random_subset_mask.shape}.",
        )

        # Validate mask sparsity.
        # With an input_count of 8 and a output_count of 600, a mean of 0.5 and variance of 0.25 is expected.
        float_mask = random_subset_mask.float()

        mean = float_mask.mean()
        self.assertTrue(
            abs(mean - 0.5) < 0.03,
            "mean of random_subset_mask must be approximately equal to 0.5\n"
            f"Received mean of {mean}.",
        )

        variance = float_mask.var()
        self.assertTrue(
            abs(variance - 0.25) < 0.03,
            "variance of random_subset_mask must be approximately equal to 0.25\n"
            f"Received a variance of {variance}.",
        )
