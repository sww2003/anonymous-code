import unittest
from unittest.mock import patch

import numpy as np
import torch

from basicsr.metrics import maniqa as maniqa_module


class _MetricRecorder:

    def __init__(self, value):
        self.value = value
        self.inputs = None

    def __call__(self, *inputs):
        self.inputs = inputs
        return torch.tensor([self.value])


class TestManiqaWrappers(unittest.TestCase):

    def test_prepare_tensor_image_converts_bgr_to_rgb(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[..., 0] = 255

        tensor = maniqa_module._prepare_tensor_image(
            image,
            crop_border=0,
            input_order='HWC',
            device=torch.device('cpu'),
            bgr2rgb=True,
        )

        self.assertEqual(tensor.shape, (1, 3, 4, 5))
        self.assertTrue(torch.all(tensor[:, 0] == 0))
        self.assertTrue(torch.all(tensor[:, 2] == 1))

    def test_maniqa_is_no_reference(self):
        recorder = _MetricRecorder(0.75)

        def replacement(metric_name, device):
            return recorder, torch.device('cpu')

        image = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch.object(
                maniqa_module, '_get_pyiqa_model', replacement):
            score = maniqa_module.calculate_maniqa(
                image,
                img2=np.ones_like(image),
                device='cpu',
            )

        self.assertAlmostEqual(score, 0.75)
        self.assertEqual(len(recorder.inputs), 1)

    def test_dists_is_full_reference(self):
        recorder = _MetricRecorder(0.25)

        def replacement(metric_name, device):
            return recorder, torch.device('cpu')

        distorted = np.zeros((8, 8, 3), dtype=np.uint8)
        reference = np.ones((8, 8, 3), dtype=np.uint8) * 255

        with patch.object(
                maniqa_module, '_get_pyiqa_model', replacement):
            score = maniqa_module.calculate_dists(
                distorted,
                reference,
                device='cpu',
            )

        self.assertAlmostEqual(score, 0.25)
        self.assertEqual(len(recorder.inputs), 2)


if __name__ == '__main__':
    unittest.main()
