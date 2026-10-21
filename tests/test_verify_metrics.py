import math
import tempfile
import unittest
from pathlib import Path

from verify_metrics import read_predictions, score


class MetricTests(unittest.TestCase):
    def test_exact_prediction(self):
        result = score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(result["rmse_deg_per_h"], 0.0)
        self.assertEqual(result["mae_deg_per_h"], 0.0)
        self.assertEqual(result["r2"], 1.0)

    def test_shared_offset_does_not_change_metrics(self):
        base = score([0.0, 1.0, 2.0], [0.1, 0.9, 2.2])
        shifted = score([8.0, 9.0, 10.0], [8.1, 8.9, 10.2])
        for key in base:
            self.assertTrue(math.isclose(base[key], shifted[key], abs_tol=1e-12))

    def test_reject_unordered_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text(
                "sample_index,target_proxy,svr,random_forest,standard_lstm,patch_transformer,timemixer,agcag\n"
                "0,0,0,0,0,0,0,0\n2,1,1,1,1,1,1,1\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                read_predictions(path)


if __name__ == "__main__":
    unittest.main()
