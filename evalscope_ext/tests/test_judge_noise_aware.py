"""Tests for the AA-LCR judge-noise-aware pruner."""

import pytest

from evalscope_ext.calibration import load_calibration
from evalscope_ext.pruners import (
    JudgeNoiseAwarePruner,
    StratifiedDiscriminativePruner,
    evaluate_selection,
    get_pruner,
    registered_strategies,
)


@pytest.fixture(scope="module")
def aa_lcr_bundle():
    return load_calibration("aa_lcr")


def test_pruner_is_registered_with_expected_name():
    assert "judge_noise_aware" in registered_strategies()
    p = get_pruner("judge_noise_aware")
    assert isinstance(p, JudgeNoiseAwarePruner)


def test_describe_includes_noise_penalty_weight():
    p = JudgeNoiseAwarePruner(noise_penalty_weight=0.5)
    desc = p.describe()
    assert desc["noise_penalty_weight"] == 0.5
    assert desc["name"] == "judge_noise_aware"


def test_rejects_invalid_noise_weight():
    with pytest.raises(ValueError):
        JudgeNoiseAwarePruner(noise_penalty_weight=-0.1)
    with pytest.raises(ValueError):
        JudgeNoiseAwarePruner(noise_penalty_weight=2.0)


def test_seed_stability_on_aa_lcr(aa_lcr_bundle):
    p = JudgeNoiseAwarePruner()
    a = p.prune(aa_lcr_bundle, keep_fraction=0.30, rng_seed=11)
    b = p.prune(aa_lcr_bundle, keep_fraction=0.30, rng_seed=11)
    assert a == b


def test_noise_penalty_zero_recovers_parent_behaviour(aa_lcr_bundle):
    """At noise_penalty_weight=0, this should select the same indices
    as the parent StratifiedDiscriminativePruner with matched defaults
    (modulo min_per_bin which we deliberately raise for AA-LCR)."""
    pj = JudgeNoiseAwarePruner(noise_penalty_weight=0.0, min_per_bin=2)
    ps = StratifiedDiscriminativePruner(min_per_bin=2)
    for seed in range(5):
        a = pj.prune(aa_lcr_bundle, keep_fraction=0.20, rng_seed=seed)
        b = ps.prune(aa_lcr_bundle, keep_fraction=0.20, rng_seed=seed)
        assert a == b, f"seed {seed}: noise-aware with weight=0 should equal parent"


def test_noise_penalty_actually_changes_output(aa_lcr_bundle):
    """With penalty > 0, at least some seeds should produce a *different*
    selection from the parent — otherwise the penalty is a no-op."""
    pj = JudgeNoiseAwarePruner(noise_penalty_weight=1.0, min_per_bin=2)
    ps = StratifiedDiscriminativePruner(min_per_bin=2)
    different = 0
    for seed in range(20):
        a = pj.prune(aa_lcr_bundle, keep_fraction=0.30, rng_seed=seed)
        b = ps.prune(aa_lcr_bundle, keep_fraction=0.30, rng_seed=seed)
        if a != b:
            different += 1
    assert different >= 5, f"only {different}/20 seeds differed — penalty appears inert"


def test_rank_correlation_holds_at_30pct_keep(aa_lcr_bundle):
    """At a 30% keep fraction, averaged across seeds, the rank correlation
    should remain >= 0.7 on AA-LCR (the threshold the compare-runs CLI
    uses as a default bar)."""
    p = JudgeNoiseAwarePruner()
    full_means = aa_lcr_bundle.per_model_mean()
    rhos = []
    for seed in range(20):
        kept = p.prune(aa_lcr_bundle, keep_fraction=0.30, rng_seed=seed)
        rhos.append(
            evaluate_selection(aa_lcr_bundle, kept).spearman_rank_correlation
        )
    assert sum(rhos) / len(rhos) >= 0.7
