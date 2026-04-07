"""
Unit tests for plx2dist.summarize_distance_posterior()
Run with: pytest tests/
"""
import math
import numpy as np
import pytest

from plx2dist import summarize_distance_posterior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(w_obs, w_err, **kwargs):
    """Thin wrapper with CNS6 defaults."""
    return summarize_distance_posterior(
        w_obs=w_obs,
        w_err=w_err,
        prior_type="edsd",
        L=250.0,
        threshold_pc=25.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Case 1: High-S/N nearby star  (σ/ϖ ≪ 1)
# A star at ~10 pc: parallax = 100 mas, σ = 0.05 mas → S/N = 2000
# ---------------------------------------------------------------------------

class TestHighSNR:
    def setup_method(self):
        self.result = _run(w_obs=100.0, w_err=0.05)

    def test_median_close_to_naive_inverse(self):
        naive = 1000.0 / 100.0  # 10 pc
        assert abs(self.result["q50"] - naive) < 0.05  # within 50 mpc

    def test_mode_close_to_median(self):
        assert abs(self.result["mode"] - self.result["q50"]) < 0.05

    def test_uncertainty_is_small(self):
        # σ/ϖ = 0.05/100 = 5e-4; relative distance error should be similar
        rel_err = self.result["err_hi"] / self.result["q50"]
        assert rel_err < 0.005  # less than 0.5%

    def test_p_within_25pc_is_unity(self):
        assert self.result["p_within_threshold_pc"] > 0.9999

    def test_all_quantiles_finite(self):
        for key in ["q05", "q16", "q50", "q84", "q95", "mode"]:
            assert math.isfinite(self.result[key]), f"{key} is not finite"

    def test_quantile_ordering(self):
        r = self.result
        assert r["q05"] < r["q16"] < r["q50"] < r["q84"] < r["q95"]


# ---------------------------------------------------------------------------
# Case 2: Low-S/N distant star  (σ/ϖ ~ 1)
# A star at ~1000 pc: parallax = 1.0 mas, σ = 1.0 mas → S/N = 1
# Prior should dominate; posterior should be well-behaved but broad.
# ---------------------------------------------------------------------------

class TestLowSNR:
    def setup_method(self):
        self.result = _run(w_obs=1.0, w_err=1.0)

    def test_median_is_positive_and_finite(self):
        assert math.isfinite(self.result["q50"])
        assert self.result["q50"] > 0.0

    def test_uncertainty_is_large(self):
        # Prior-dominated: relative uncertainty should be >> 10%
        rel_err = self.result["err_hi"] / self.result["q50"]
        assert rel_err > 0.1

    def test_quantile_ordering(self):
        r = self.result
        assert r["q05"] < r["q16"] < r["q50"] < r["q84"] < r["q95"]

    def test_all_outputs_finite(self):
        for key in ["q05", "q16", "q50", "q84", "q95", "mode",
                    "err_lo", "err_hi", "p_within_threshold_pc"]:
            assert math.isfinite(self.result[key]), f"{key} is not finite"

    def test_p_within_25pc_is_less_than_one(self):
        # Prior-dominated distant star: should NOT be certain to be within 25 pc
        assert self.result["p_within_threshold_pc"] < 0.99


# ---------------------------------------------------------------------------
# Case 3: Negative parallax  (physically valid measurement, common in Gaia)
# Posterior should still be well-behaved due to prior regularisation.
# ---------------------------------------------------------------------------

class TestNegativeParallax:
    def setup_method(self):
        self.result = _run(w_obs=-2.0, w_err=1.5)

    def test_does_not_raise(self):
        # The function must return without exception
        assert isinstance(self.result, dict)

    def test_median_is_positive_and_finite(self):
        assert math.isfinite(self.result["q50"])
        assert self.result["q50"] > 0.0

    def test_prior_dominated_broad_posterior(self):
        # With negative parallax the posterior is essentially the prior;
        # expect a very large q84 relative to q16
        assert self.result["q84"] > 5.0 * self.result["q16"]

    def test_p_within_25pc_is_small(self):
        # Negative parallax → almost certainly not nearby
        assert self.result["p_within_threshold_pc"] < 0.2


# ---------------------------------------------------------------------------
# Case 4: Zero parallax  (degenerate but must not crash)
# ---------------------------------------------------------------------------

class TestZeroParallax:
    def test_prior_dominated(self):
        r = _run(w_obs=0.0, w_err=0.5)
        assert math.isfinite(r["q50"]) and r["q50"] > 0.0
        assert r["p_within_threshold_pc"] < 1e-3


# ---------------------------------------------------------------------------
# Case 5: Masked parallax  (numpy masked array elements)
# ---------------------------------------------------------------------------

class TestMaskedParallax:
    def test_masked_w_obs_raises(self):
        w_masked = np.ma.masked
        with pytest.raises((ValueError, TypeError)):
            _run(w_obs=w_masked, w_err=0.5)

    def test_masked_w_err_raises(self):
        e_masked = np.ma.masked
        with pytest.raises((ValueError, TypeError)):
            _run(w_obs=50.0, w_err=e_masked)


# ---------------------------------------------------------------------------
# Case 6: Volume prior sanity check
# High-S/N star — volume prior should give similar median to EDSD prior
# ---------------------------------------------------------------------------

class TestVolumePrior:
    def test_volume_prior_median_close_to_edsd(self):
        edsd = summarize_distance_posterior(
            100.0, 0.05, prior_type="edsd", L=250.0, threshold_pc=25.0
        )
        vol = summarize_distance_posterior(
            100.0, 0.05, prior_type="volume", threshold_pc=25.0
        )
        # High-S/N: likelihood dominates, prior choice barely matters
        assert abs(edsd["q50"] - vol["q50"]) < 0.1  # pc


# ---------------------------------------------------------------------------
# Case 7: Regression — grid outputs are deterministic (no randomness)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        r1 = _run(w_obs=40.0, w_err=0.3)
        r2 = _run(w_obs=40.0, w_err=0.3)
        assert r1["q50"] == r2["q50"]
        assert r1["mode"] == r2["mode"]