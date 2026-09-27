"""Checks the Hearsay minDCF configuration and condition-router contract."""

from __future__ import annotations

import unittest

import numpy as np

from hearsay.routing import fit_router, min_dcf


class RoutingTest(unittest.TestCase):
    def test_min_dcf_uses_hearsay_costs(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(min_dcf(labels, perfect), (0.0, 0.8))

    def test_router_returns_bounded_blend_scores(self) -> None:
        rng = np.random.default_rng(27)
        labels = np.tile([0, 1], 60)
        quality = rng.normal(size=(len(labels), 12))
        # Two recording conditions are independent of the target label.
        quality[:, 0] = np.repeat([0.0, 2.0], 60)
        arshiya = np.clip(0.25 + 0.5 * labels + rng.normal(0, 0.08, len(labels)), 0, 1)
        julia = np.clip(0.35 + 0.4 * labels + rng.normal(0, 0.08, len(labels)), 0, 1)
        router = fit_router(quality, labels, arshiya, julia, clusters=2, minimum_cluster_size=20)
        result = router.predict(arshiya[:8], julia[:8], quality[:8])
        self.assertEqual(result.shape, (8,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0) & (result <= 1)))


if __name__ == "__main__":
    unittest.main()
