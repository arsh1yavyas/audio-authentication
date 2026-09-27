"""Checks the Hearsay minDCF configuration and condition-router contract."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from hearsay.routing import dcf_at_cutoff, fit_router, min_dcf


class RoutingTest(unittest.TestCase):
    def test_min_dcf_uses_hearsay_costs(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(min_dcf(labels, perfect), (0.0, 0.8))

        # Rejecting one of ten real clips costs 2.8 * 0.1 / 0.3, even when
        # every spoof clip is caught. Accepting all clips costs exactly 1.
        labels = np.asarray([0] * 10 + [1] * 10)
        scores = np.asarray([0.1] * 9 + [0.9] + [0.8] * 10)
        cost, threshold = min_dcf(labels, scores)
        self.assertAlmostEqual(cost, 2.8 * 0.1 / 0.3)
        self.assertEqual(threshold, 0.8)

    def test_min_dcf_handles_tied_scores_and_extreme_decisions(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        tied = np.asarray([0.1, 0.5, 0.5, 0.9])
        self.assertEqual(min_dcf(labels, tied), (0.5, 0.9))

        identical = np.full(4, 0.5)
        cost, threshold = min_dcf(labels, identical)
        self.assertEqual(cost, 1.0)
        self.assertGreater(threshold, 0.5)  # Every clip is accepted as real.
        cost, threshold = min_dcf(labels, identical, c_miss=4.0, c_fa=1.0)
        self.assertEqual(cost, 1.0)
        self.assertEqual(threshold, 0.5)  # Every clip is rejected as spoof.

    def test_positive_score_shrink_preserves_mindcf_and_ranking(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scores = np.asarray([0.12, 0.48, 0.73, 0.41, 0.82, 0.95])
        shrunk = scores * 0.05
        self.assertEqual(min_dcf(labels, scores)[0], min_dcf(labels, shrunk)[0])
        self.assertEqual(roc_auc_score(labels, scores), roc_auc_score(labels, shrunk))

    def test_min_dcf_rejects_nonpositive_costs(self) -> None:
        labels = np.asarray([0, 1])
        scores = np.asarray([0.1, 0.9])
        with self.assertRaisesRegex(ValueError, "costs must be finite and positive"):
            min_dcf(labels, scores, c_fa=0.0)

    def test_fixed_cutoff_uses_same_decisions_and_costs_as_min_dcf(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scores = np.asarray([0.2, 0.5, 0.8, 0.5, 0.7, 0.9])
        cost, cutoff = min_dcf(labels, scores)
        self.assertAlmostEqual(dcf_at_cutoff(labels, scores, cutoff), cost)
        self.assertAlmostEqual(dcf_at_cutoff(labels, scores, 0.5),
                               (2.8 * (2 / 3) + 0.3 * 0) / 0.3)
        self.assertEqual(dcf_at_cutoff(labels, scores, np.inf), 1.0)
        self.assertAlmostEqual(dcf_at_cutoff(labels, scores, -np.inf), 2.8 / 0.3)
        self.assertAlmostEqual(dcf_at_cutoff(labels, scores, 0.5,
                                             p_spoof=0.5, c_miss=2.0, c_fa=1.0),
                               (0.5 * (2 / 3)) / 0.5)

    def test_fixed_cutoff_rejects_nan_cutoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cutoff must be"):
            dcf_at_cutoff(np.asarray([0, 1]), np.asarray([0.1, 0.9]), np.nan)

    def test_router_returns_bounded_blend_scores(self) -> None:
        rng = np.random.default_rng(27)
        labels = np.tile([0, 1], 60)
        quality = rng.normal(size=(len(labels), 12))
        # Two recording conditions are independent of the target label.
        quality[:, 0] = np.repeat([0.0, 2.0], 60)
        arshiya = np.clip(0.25 + 0.5 * labels + rng.normal(0, 0.08, len(labels)), 0, 1)
        julia = np.clip(0.35 + 0.4 * labels + rng.normal(0, 0.08, len(labels)), 0, 1)
        router = fit_router(quality, labels, arshiya, julia, clusters=2, minimum_cluster_size=20)
        result = router.predict(arshiya[:8], julia[:8], quality[:8].astype(np.float32))
        self.assertEqual(result.shape, (8,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0) & (result <= 1)))


if __name__ == "__main__":
    unittest.main()
