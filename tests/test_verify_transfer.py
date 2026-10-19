import math
import tempfile
import unittest
from pathlib import Path

from verify_transfer import calculate, read_transfer


class TransferTests(unittest.TestCase):
    def test_synthetic_scores_and_axis_mean(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_transfer.csv"
        rows = calculate(read_transfer(path))
        self.assertEqual(len(rows), 24)
        overall = next(
            item for item in rows
            if item[0:3] == ("A_to_B", "mean_of_axes", "all")
        )
        self.assertEqual(overall[3], 12)
        self.assertTrue(math.isclose(overall[4]["rmse_deg_per_h"], 0.01))

    def test_reject_duplicate_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text(
                "direction,axis,thermal_segment,sample_index,target_proxy,agcag\n"
                "A_to_B,X,dynamic,0,0.0,0.1\n"
                "A_to_B,X,dynamic,0,0.1,0.0\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                read_transfer(path)


if __name__ == "__main__":
    unittest.main()
