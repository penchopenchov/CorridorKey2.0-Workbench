import unittest
import numpy as np
import torch
from corridorkey import color_utils as cu

class ColorFastTests(unittest.TestCase):
    def test_nonnegative_transfer_preserves_float_bits_at_knee_and_hdr(self):
        self.assertTrue(hasattr(cu, 'linear_to_srgb_nonnegative'), 'nonnegative transfer fast path is missing')
        rng = np.random.default_rng(31)
        values = rng.uniform(0, 32, (17, 3, 29)).astype(np.float32)
        knee = np.float32(.0031308)
        values.flat[:8] = [0., -0., knee, np.nextafter(knee,np.float32(0)),
                          np.nextafter(knee,np.float32(1)), 1., np.inf, .04045]
        for value in (values, torch.from_numpy(values)):
            old = cu.linear_to_srgb(value)
            fast = cu.linear_to_srgb_nonnegative(value)
            if isinstance(old,torch.Tensor):
                old,fast=old.numpy(),fast.numpy()
            np.testing.assert_array_equal(old.view(np.uint32),fast.view(np.uint32))
        if torch.cuda.is_available():
            value = torch.from_numpy(values).cuda()
            old=cu.linear_to_srgb(value).cpu().numpy()
            fast=cu.linear_to_srgb_nonnegative(value).cpu().numpy()
            np.testing.assert_array_equal(old.view(np.uint32),fast.view(np.uint32))
