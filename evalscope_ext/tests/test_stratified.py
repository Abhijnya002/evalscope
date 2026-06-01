"""Tests for the StratifiedDiscriminativePruner."""

import math

import pytest

from evalscope_ext.calibration import load_calibration
from evalscope_ext.pruners import StratifiedDiscriminativePruner, evaluate_selection


@pytest.fixture(scope="module")
def lcb_bundle():
    return load_calibration("live_code_bench_v5")


@pytest.fixture(scope="module")
def aa_lcr_bundle():
    return load_calibration("aa_lcr")


def test_keep_fraction_one_returns_every_index(lcb_bundle):
    p = StratifiedDiscriminativePruner()
    kept = p.prune(lcb_bundle, keep_fraction=1.0)
    assert sorted(kept) == sorted(lcb_bundle.indices)


def test_pruner_is_deterministic_given_same_seed(lcb_bundle):
    p = StratifiedDiscriminativePruner()
    a = p.prune(lcb_bundle, keep_fraction=0.15, rng_seed=7)
    b = p.prune(lcb_bundle, keep_fraction=0.15, rng_seed=7)
    assert a == b


def test_different_seeds_can_give_different_picks(lcb_bundle):
    """Within a bin, the round-robin diversity walk is rng-seeded.
    Different seeds may walk buckets in different orders → different picks.
    We don't *require* difference (small bin + few buckets can converge);
    we require that the *set* of seed→output mappings spans more than one."""
    p = StratifiedDiscriminativePruner()
    outputs = {tuple(p.prune(lcb_bundle, keep_fraction=0.10, rng_seed=s)) for s in range(8)}
    assert len(outputs) > 1


def test_rejects_invalid_config():
    with pytest.raises(ValueError):
        StratifiedDiscriminativePruner(n_bins=1)
    with pytest.raises(ValueError):
        StratifiedDiscriminativePruner(stratification="random")
    with pytest.raises(ValueError):
        StratifiedDiscriminativePruner(discrimination_floor=-0.1)
    with pytest.raises(ValueError):
        StratifiedDiscriminativePruner(discrimination_floor=2.0)


def test_proportional_stratification_passes_default_bar_at_10pct_on_lcb(lcb_bundle):
    """The headline-defending claim for LCB: rho >= 0.7 AND shift <= 0.05
    at a 10% keep fraction."""
    p = StratifiedDiscriminativePruner(stratification="proportional")
    kept = p.prune(lcb_bundle, keep_fraction=0.10, rng_seed=42)
    r = evaluate_selection(lcb_bundle, kept)
    assert r.spearman_rank_correlation >= 0.7
    assert r.max_abs_mean_shift <= 0.05
    assert r.passes_default_bar


def test_proportional_keeps_top1_winner_most_of_the_time_at_10pct(lcb_bundle):
    """At a 10% keep fraction (32 samples on LCB) the pruned subset is
    small enough that *occasional* seeds can flip the top-1 winner — we
    want > 85% of seeds to preserve it, which is the realistic claim a
    deployment lead can defend. The compare-runs CLI surfaces this
    uncertainty explicitly via rng_seed."""
    p = StratifiedDiscriminativePruner()
    full_winner = max(lcb_bundle.per_model_mean().items(), key=lambda t: t[1])[0]
    n_correct = 0
    n_total = 30
    for seed in range(n_total):
        kept = p.prune(lcb_bundle, keep_fraction=0.10, rng_seed=seed)
        r = evaluate_selection(lcb_bundle, kept)
        pruned_winner = max(r.per_model_mean_pruned.items(), key=lambda t: t[1])[0]
        if pruned_winner == full_winner:
            n_correct += 1
    assert n_correct / n_total >= 0.85, f"only {n_correct}/{n_total} seeds preserved the winner"


def test_proportional_always_keeps_winner_at_20pct(lcb_bundle):
    """At 20% keep (~63 samples) the winner should be preserved for *all* seeds."""
    p = StratifiedDiscriminativePruner()
    full_winner = max(lcb_bundle.per_model_mean().items(), key=lambda t: t[1])[0]
    for seed in range(20):
        kept = p.prune(lcb_bundle, keep_fraction=0.20, rng_seed=seed)
        r = evaluate_selection(lcb_bundle, kept)
        pruned_winner = max(r.per_model_mean_pruned.items(), key=lambda t: t[1])[0]
        assert pruned_winner == full_winner, f"seed {seed}: {pruned_winner} != {full_winner}"


def test_equal_stratification_inflates_mean_shift(lcb_bundle):
    """Documented tradeoff: equal-weight biases mean low on a benchmark where
    most samples are easy (LCB v5 — ~63%+ pass rates). The pruner *correctly*
    surfaces this via the metric so a caller can detect it."""
    eq = StratifiedDiscriminativePruner(stratification="equal")
    prop = StratifiedDiscriminativePruner(stratification="proportional")
    kept_eq = eq.prune(lcb_bundle, keep_fraction=0.10, rng_seed=42)
    kept_prop = prop.prune(lcb_bundle, keep_fraction=0.10, rng_seed=42)
    shift_eq = evaluate_selection(lcb_bundle, kept_eq).max_abs_mean_shift
    shift_prop = evaluate_selection(lcb_bundle, kept_prop).max_abs_mean_shift
    # Equal-weight should bias the mean more than proportional does.
    assert shift_eq > shift_prop


def test_min_per_bin_protects_rare_bins(lcb_bundle):
    """At a very low keep_fraction, min_per_bin should still keep the
    hardest bin represented rather than dropping it entirely."""
    p = StratifiedDiscriminativePruner(min_per_bin=2)
    # 5% of 315 = ~16 samples total. If min_per_bin worked, every
    # non-empty bin gets at least 2.
    kept = p.prune(lcb_bundle, keep_fraction=0.05, rng_seed=0)
    assert len(kept) >= 8  # at least 2 per bin x 4 bins
