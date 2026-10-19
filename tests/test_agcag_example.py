import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from agcag_example import AGCAG


@unittest.skipIf(torch is None, "PyTorch is optional for the metric-only tests")
class ArchitectureTests(unittest.TestCase):
    def test_shape_finite_and_gradients(self):
        model = AGCAG(input_dim=24, hidden_dim=32)
        prediction = model(torch.randn(2, 20, 24))
        self.assertEqual(tuple(prediction.shape), (2, 1))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        prediction.sum().backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)

    def test_reject_wrong_feature_count(self):
        model = AGCAG(input_dim=24, hidden_dim=32)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 20, 21))


if __name__ == "__main__":
    unittest.main()
