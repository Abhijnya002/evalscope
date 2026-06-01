"""Tests for the calibration loader against the vendored slim data."""

import math

import pytest

from evalscope_ext.calibration import (
    CalibrationBundle,
    difficulty_bins,
    load_calibration,
)


def test_lcb_loads_315_samples_across_three_models():
    b = load_calibration("live_code_bench_v5")
    assert isinstance(b, CalibrationBundle)
    assert b.n_samples == 315
    assert set(b.models) == {"gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"}
    # Every sample should have scores for all three reference models.
    for s in b.samples:
        assert set(s.scores) == set(b.models)


def test_aa_lcr_loads_100_samples():
    b = load_calibration("aa_lcr")
    assert b.n_samples == 100
    assert set(b.models) == {"gpt-oss-120b", "kimi-k2.5", "minimax-m2.5"}


def test_lcb_per_model_means_match_known_pass_rates():
    """Sanity-check the loader produced the right pass rates.

    Numbers from the shipped data:
      gpt-oss-120b  241/315 = 0.7651
      kimi-k2.5     198/315 = 0.6286
      minimax-m2.5  195/315 = 0.6190
    """
    b = load_calibration("live_code_bench_v5")
    means = b.per_model_mean()
    assert math.isclose(means["gpt-oss-120b"], 241 / 315, abs_tol=1e-3)
    assert math.isclose(means["kimi-k2.5"], 198 / 315, abs_tol=1e-3)
    assert math.isclose(means["minimax-m2.5"], 195 / 315, abs_tol=1e-3)


def test_discrimination_is_zero_when_all_models_agree():
    from evalscope_ext.calibration import SampleScores

    s = SampleScores(index=0, scores={"a": 1.0, "b": 1.0, "c": 1.0}, metadata={})
    assert s.discrimination == 0.0
    s = SampleScores(index=1, scores={"a": 0.0, "b": 0.0, "c": 0.0}, metadata={})
    assert s.discrimination == 0.0


def test_discrimination_is_maximal_when_models_split():
    from evalscope_ext.calibration import SampleScores

    # 2 pass + 2 fail → p=0.5 → discrimination = 1.0 (after rescaling)
    s = SampleScores(
        index=2,
        scores={"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0},
        metadata={},
    )
    assert math.isclose(s.discrimination, 1.0, abs_tol=1e-6)


def test_difficulty_bins_partition_all_samples():
    b = load_calibration("live_code_bench_v5")
    bins = difficulty_bins(b.samples, n_bins=4)
    assert sum(len(v) for v in bins.values()) == b.n_samples
    # Bin 0 should contain samples with the lowest pass fractions.
    if bins[0]:
        assert max(s.pass_fraction for s in bins[0]) < 0.25
    if bins[3]:
        # Highest bin includes pass_fraction == 1.0
        assert min(s.pass_fraction for s in bins[3]) >= 0.75


def test_aa_lcr_per_model_means_match_known_acc():
    """gpt-oss-120b 48/100, kimi-k2.5 66/100, minimax-m2.5 64/100."""
    b = load_calibration("aa_lcr")
    means = b.per_model_mean()
    assert math.isclose(means["gpt-oss-120b"], 0.48, abs_tol=1e-3)
    assert math.isclose(means["kimi-k2.5"], 0.66, abs_tol=1e-3)
    assert math.isclose(means["minimax-m2.5"], 0.64, abs_tol=1e-3)


def test_missing_benchmark_raises():
    with pytest.raises(FileNotFoundError):
        load_calibration("not_a_real_benchmark")
