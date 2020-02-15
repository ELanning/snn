import unittest
import torch as t
from black_sheep.weight import normal_weight_init

arbitrary_number = 333
t.manual_seed(arbitrary_number)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False


class TestWeight(unittest.TestCase):
    def test_normal_weight_init(self):
        input_count = 800
        output_count = 400
        normal_weights = normal_weight_init(
            input_count, output_count, connection_count=output_count
        )

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
            abs(mean) < 0.01,
            "mean of normal_weights must be approximately equal to zero.\n"
            f"Received mean of {mean}",
        )

        variance = normal_weights.var()
        self.assertTrue(
            abs(variance - 1) < 0.01,
            "variance of normal_weights must be approximately equal to one.\n"
            f"Received variance of {variance}",
        )
