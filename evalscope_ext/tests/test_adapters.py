"""Tests for the pruned dataset adapters.

These tests assert *registration + wiring* — that the
``@register_benchmark`` decorators run on import, the
``--dataset-args`` parameters are declared, and an instantiated
adapter resolves the pruner correctly.

They don't run a real ``evalscope eval`` pipeline — that requires
downloading the upstream dataset + invoking a live model, which is out
of scope for unit tests. The compare_runs CLI's smoke test
(``test_compare_runs.py``) covers the post-run path against synthetic
prediction jsonl.
"""

from __future__ import annotations

import pytest

pytest.importorskip("evalscope")  # skip the whole module without evalscope

import evalscope_ext.adapters  # noqa: F401 — side-imports register the benchmarks
from evalscope.api.registry import BENCHMARK_REGISTRY


def _get_meta_and_cls(name: str):
    """Return the (meta, adapter_cls) tuple from the registry.

    evalscope's registry shape has shifted across releases; this helper
    abstracts the differences so the tests don't break with an upstream
    bump.
    """
    entry = BENCHMARK_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"benchmark {name!r} not registered")
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry  # (meta, cls)
    # Fall back: registry stores BenchmarkMeta objects with a data_adapter attribute.
    meta = entry
    cls = getattr(meta, "data_adapter", None) or getattr(meta, "adapter_cls", None)
    return meta, cls


def test_live_code_bench_pruned_is_registered():
    meta, cls = _get_meta_and_cls("live_code_bench_pruned")
    assert cls is not None
    assert cls.__name__ == "LiveCodeBenchPrunedAdapter"
    # Must inherit from the upstream LCB adapter so all its eval logic
    # (sandboxed code execution, judge / scoring) still works.
    from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

    assert issubclass(cls, LiveCodeBenchAdapter)


def test_aa_lcr_pruned_is_registered():
    meta, cls = _get_meta_and_cls("aa_lcr_pruned")
    assert cls is not None
    assert cls.__name__ == "AaLcrPrunedAdapter"
    from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter

    assert issubclass(cls, AALCRAdapter)


@pytest.mark.parametrize("name", ["live_code_bench_pruned", "aa_lcr_pruned"])
def test_extra_params_carry_pruning_knobs(name):
    meta, _ = _get_meta_and_cls(name)
    keys = set(meta.extra_params.keys())
    # The four parameters the run contract specifies.
    for required in ("pruning_strategy", "prune_ratio", "rng_seed", "calibration_benchmark"):
        assert required in keys, f"{name} missing extra_param: {required}"


def test_lcb_pruned_default_strategy_is_stratified_discriminative():
    meta, _ = _get_meta_and_cls("live_code_bench_pruned")
    p = meta.extra_params["pruning_strategy"]
    default = p["value"] if isinstance(p, dict) else p
    assert default == "stratified_discriminative"


def test_aa_lcr_pruned_default_strategy_is_judge_noise_aware():
    """AA-LCR should default to the judge-noise-aware strategy because
    the benchmark is LLM-judged. This is a defensible-by-default
    choice, not a hidden one."""
    meta, _ = _get_meta_and_cls("aa_lcr_pruned")
    p = meta.extra_params["pruning_strategy"]
    default = p["value"] if isinstance(p, dict) else p
    assert default == "judge_noise_aware"
