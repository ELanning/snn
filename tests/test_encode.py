import unittest
import torch as t
from black_sheep.encode import poisson_encode

arbitrary_number = 333
t.manual_seed(arbitrary_number)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False


class TestEncode(unittest.TestCase):
    def test_poisson_encode(self):
        example = t.empty(3, 3).uniform_(0, 1)
        # A reasonably high number was chosen so that the decoder can properly reconstruct the input.
        time_slice_count = 100
        encoding = poisson_encode(example, spike_train_count=time_slice_count)

        # Validate output dimension.
        expected_dimension = (example.numel(), time_slice_count)
        self.assertEqual(expected_dimension, encoding.size(), "Encoding dimension was incorrect.")

        # Validate decoding back to the original example.
        decoding = encoding.mean(dim=1).view(example.size())
        approximately_close = t.all(t.abs(example - decoding) < 0.3)
        self.assertTrue(approximately_close, "Decoding was not close enough to the original.")

        # Check Fano Factor, which can be used to measure if a process is Poisson.
        histogram = t.histc(encoding.sum(dim=1))
        fano_factor = histogram.var() / histogram.mean()
        self.assertTrue(abs(fano_factor - 1) < 0.1, "Fano factor was not close enough to one.")

    def test_poisson_encode_bad_inputs(self):
        negative_spike_train_count = -1
        empty_spike_train_count = 0

        self.assertRaises(ValueError, poisson_encode, t.empty(1), negative_spike_train_count)
        self.assertRaises(ValueError, poisson_encode, t.empty(1), empty_spike_train_count)
