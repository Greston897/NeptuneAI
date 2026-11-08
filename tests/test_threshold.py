import unittest

from training.threshold import binary_metrics, select_threshold


class ThresholdTests(unittest.TestCase):
    def test_chooses_recall_maximizing_threshold_that_meets_precision(self):
        threshold, metrics = select_threshold(
            [0.98, 0.93, 0.87, 0.52], [1, 1, 0, 0], target_precision=0.95
        )
        self.assertLessEqual(threshold, 0.93)
        self.assertGreaterEqual(metrics["precision"], 0.95)
        self.assertEqual(metrics["recall"], 1.0)

    def test_falls_back_to_conservative_threshold_when_target_impossible(self):
        threshold, metrics = select_threshold([0.8, 0.9], [0, 0], target_precision=0.98)
        self.assertEqual(threshold, 0.99)
        self.assertEqual(metrics["recall"], 0.0)

    def test_binary_metrics_match_expected_values(self):
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_rejects_empty_or_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            select_threshold([], [], 0.98)
        with self.assertRaises(ValueError):
            select_threshold([0.4], [0, 1], 0.98)


if __name__ == "__main__":
    unittest.main()
