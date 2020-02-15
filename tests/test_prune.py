import unittest
import torch as t
from black_sheep.prune import random_subset_mask_init

arbitrary_number = 333
t.manual_seed(arbitrary_number)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False


class TestPrune(unittest.TestCase):
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
