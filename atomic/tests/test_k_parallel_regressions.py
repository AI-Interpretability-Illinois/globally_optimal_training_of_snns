"""Regression checks for K_parallel manifest validation."""

from __future__ import annotations

import unittest

from atomic.finetune_manifest import assert_finetune_manifest_matches_runtime

_BASE_MAN = {
    "L": 3,
    "P_rec": 4,
    "P_last": 8,
    "P_in": 5,
    "num_classes": 3,
    "readout_mode": "membrane",
}


class TestFinetuneManifestKParallel(unittest.TestCase):
    def test_legacy_missing_k_allows_only_k1(self) -> None:
        assert_finetune_manifest_matches_runtime(
            dict(_BASE_MAN),
            readout_mode="membrane",
            P_in=5,
            P_rec=4,
            P_last=8,
            L=3,
            num_classes=3,
            K_parallel=1,
        )
        with self.assertRaises(ValueError):
            assert_finetune_manifest_matches_runtime(
                dict(_BASE_MAN),
                readout_mode="membrane",
                P_in=5,
                P_rec=4,
                P_last=8,
                L=3,
                num_classes=3,
                K_parallel=2,
            )

    def test_stored_k_must_match_runtime(self) -> None:
        m = {**_BASE_MAN, "K_parallel": 2}
        assert_finetune_manifest_matches_runtime(
            m,
            readout_mode="membrane",
            P_in=5,
            P_rec=4,
            P_last=8,
            L=3,
            num_classes=3,
            K_parallel=2,
        )
        with self.assertRaises(ValueError):
            assert_finetune_manifest_matches_runtime(
                m,
                readout_mode="membrane",
                P_in=5,
                P_rec=4,
                P_last=8,
                L=3,
                num_classes=3,
                K_parallel=3,
            )


if __name__ == "__main__":
    unittest.main()
