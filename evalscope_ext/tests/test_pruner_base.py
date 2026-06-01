"""Tests for the Pruner ABC + selection-quality metrics."""

import math

import pytest

from evalscope_ext.calibration import load_calibration
from evalscope_ext.pruners import (
    Pruner,
    evaluate_selection,
    get_pruner,
    register_pruner,
    registered_strategies,
    validate_keep_fraction,
)


def test_validate_keep_fraction_accepts_open_interval():
    validate_keep_fraction(0.1)
    validate_keep_fraction(1.0)
    with pytest.raises(ValueError):
        validate_keep_fraction(0.0)
    with pytest.raises(ValueError):
        validate_keep_fraction(1.5)


def test_unknown_pruner_raises():
    with pytest.raises(KeyError):
        get_pruner("not_a_real_strategy")


def test_register_pruner_decorator_adds_to_registry():
    @register_pruner
    class _Trivial(Pruner):
        name = "test_trivial_pruner"

        def prune(self, bundle, *, keep_fraction, rng_seed=0):
            n = max(1, int(round(bundle.n_samples * keep_fraction)))
            return bundle.indices[:n]

    assert "test_trivial_pruner" in registered_strategies()
    p = get_pruner("test_trivial_pruner")
    bundle = load_calibration("live_code_bench_v5")
    kept = p.prune(bundle, keep_fraction=0.1)
    assert len(kept) == round(bundle.n_samples * 0.1)


def test_register_rejects_duplicate_name():
    @register_pruner
    class _A(Pruner):
        name = "test_duplicate_check"

        def prune(self, bundle, *, keep_fraction, rng_seed=0):
            return list(bundle.indices)

    with pytest.raises(ValueError):

        @register_pruner
        class _B(Pruner):
            name = "test_duplicate_check"  # same name

            def prune(self, bundle, *, keep_fraction, rng_seed=0):
                return list(bundle.indices)


def test_evaluate_selection_perfect_when_keeping_everything():
    bundle = load_calibration("live_code_bench_v5")
    report = evaluate_selection(bundle, bundle.indices)
    assert report.pruned_size == bundle.n_samples
    assert math.isclose(report.spearman_rank_correlation, 1.0, abs_tol=1e-9)
    assert report.max_abs_mean_shift == 0.0
    assert all(v == 1.0 for v in report.top_k_retention.values())


def test_evaluate_selection_detects_bad_subset():
    """If we hand-pick the top-K easiest samples, the mean shift should
    be huge and the rank correlation may collapse — which is exactly
    the kind of thing the rubric flags as a 'forbidden trivial
    baseline'. The metric should agree."""
    bundle = load_calibration("live_code_bench_v5")
    easy_indices = [s.index for s in sorted(bundle.samples, key=lambda s: -s.pass_fraction)[:30]]
    report = evaluate_selection(bundle, easy_indices)
    # All-easy subset means almost everyone passes — mean shift is large.
    assert report.max_abs_mean_shift > 0.1


def test_evaluate_selection_raises_on_unknown_index():
    bundle = load_calibration("live_code_bench_v5")
    with pytest.raises(KeyError):
        evaluate_selection(bundle, [-1, -2, -3])
