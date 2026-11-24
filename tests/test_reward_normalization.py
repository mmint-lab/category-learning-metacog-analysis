"""Checks against independent mathematical bounds, not the legacy denominator."""
import itertools
import unittest

import numpy as np

from analysisfunctions import calcCalibration, convertCalibration


class RewardNormalizationTests(unittest.TestCase):
    def test_correct_low_confidence_is_not_worst_possible(self):
        # For f=9/16, errors are 49/256 if correct and 81/256 if wrong.
        score = convertCalibration(np.ones(42, dtype=int), np.ones(42))
        self.assertAlmostEqual(score, 32/81)

    def test_all_incorrect_achieves_zero_for_unequal_bin_counts(self):
        keys = np.array([1]*9 + [2]*5 + [3]*3 + [4])
        self.assertAlmostEqual(convertCalibration(keys, np.zeros(len(keys))), 0)

    def test_perfect_bin_calibration_achieves_one(self):
        keys = np.repeat(np.arange(1, 5), 16)
        outcomes = np.concatenate([np.r_[np.ones(k), np.zeros(16-k)] for k in [9, 11, 13, 15]])
        self.assertAlmostEqual(convertCalibration(keys, outcomes), 1)

    def test_exhaustive_outcomes_match_fixed_distribution_maximum(self):
        keys = np.array([1, 1, 2, 2, 3, 4])
        probabilities = np.array([9, 9, 11, 11, 13, 15])/16
        combinations = list(itertools.product([0, 1], repeat=len(keys)))
        errors = [calcCalibration(keys, np.array(c)) for c in combinations]
        maximum = max(errors)
        self.assertAlmostEqual(maximum, np.mean(probabilities**2))
        for outcomes, error in zip(combinations, errors):
            score = convertCalibration(keys, np.array(outcomes))
            self.assertAlmostEqual(score, 1-error/maximum)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_uniform_export_replication_preserves_score(self):
        keys = np.array([1, 1, 1, 2, 3, 4])
        outcomes = np.array([1, 0, 1, 0, 1, 1])
        self.assertAlmostEqual(convertCalibration(keys, outcomes),
                               convertCalibration(np.repeat(keys, 4), np.repeat(outcomes, 4)))


if __name__ == '__main__':
    unittest.main()
