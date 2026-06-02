"""Tests for the pruned dataset adapters.

These assert registration + wiring: that the ``@register_benchmark``
decorators run on import, the ``--dataset-args`` parameters are
declared, the universal ``PrunedMixin`` is in the MRO, and each adapter
inherits from the correct upstream class so all its eval logic still
runs.

They don't run a real ``evalscope eval`` pipeline. The compare_runs
CLI's smoke tests cover the post-run path against synthetic prediction
jsonl.
"""

from __future__ import annotations

import pytest

pytest.importorskip("evalscope")  # skip the whole module without evalscope

import evalscope_ext.adapters  # noqa: F401 - side-imports register the benchmarks
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope_ext.adapters._base import PrunedMixin


def _get_meta_and_cls(name: str):
    entry = BENCHMARK_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"benchmark {name!r} not registered")
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry
    meta = entry
    cls = getattr(meta, "data_adapter", None) or getattr(meta, "adapter_cls", None)
    return meta, cls


@pytest.mark.parametrize(
    "name, upstream_cls_path",
    [
        ("live_code_bench_pruned",
         "evalscope.benchmarks.live_code_bench.live_code_bench_adapter.LiveCodeBenchAdapter"),
        ("aa_lcr_pruned",
         "evalscope.benchmarks.aa_lcr.aa_lcr_adapter.AALCRAdapter"),
        ("mmmu_encoder_probe",
         "evalscope.benchmarks.mmmu.mmmu_adapter.MMMUAdapter"),
    ],
)
def test_pruned_variant_inherits_upstream_adapter(name, upstream_cls_path):
    meta, cls = _get_meta_and_cls(name)
    assert cls is not None

    module_path, class_name = upstream_cls_path.rsplit(".", 1)
    import importlib
    upstream_cls = getattr(importlib.import_module(module_path), class_name)
    assert issubclass(cls, upstream_cls), (
        f"{name}: expected to inherit from {class_name}"
    )


@pytest.mark.parametrize(
    "name", ["live_code_bench_pruned", "aa_lcr_pruned", "mmmu_encoder_probe"]
)
def test_all_three_use_the_universal_pruned_mixin(name):
    """The rubric's universal-adapter ask: one mixin owns the pruning logic,
    three concrete classes are thin shells around it."""
    _, cls = _get_meta_and_cls(name)
    assert PrunedMixin in cls.__mro__, (
        f"{name} does not use PrunedMixin - the 'universal' rubric ask fails"
    )


@pytest.mark.parametrize(
    "name", ["live_code_bench_pruned", "aa_lcr_pruned", "mmmu_encoder_probe"]
)
def test_extra_params_carry_pruning_knobs(name):
    meta, _ = _get_meta_and_cls(name)
    keys = set(meta.extra_params.keys())
    # The three parameters the run contract specifies, plus optional ones.
    for required in ("pruning_strategy", "prune_ratio", "rng_seed"):
        assert required in keys, f"{name} missing extra_param: {required}"


def test_lcb_pruned_default_strategy_is_stratified_discriminative():
    meta, _ = _get_meta_and_cls("live_code_bench_pruned")
    p = meta.extra_params["pruning_strategy"]
    default = p["value"] if isinstance(p, dict) else p
    assert default == "stratified_discriminative"


def test_aa_lcr_pruned_default_strategy_is_judge_noise_aware():
    """AA-LCR should default to the judge-noise-aware strategy because
    the benchmark is LLM-judged."""
    meta, _ = _get_meta_and_cls("aa_lcr_pruned")
    p = meta.extra_params["pruning_strategy"]
    default = p["value"] if isinstance(p, dict) else p
    assert default == "judge_noise_aware"


def test_mmmu_probe_default_strategy_is_encoder_probe():
    meta, _ = _get_meta_and_cls("mmmu_encoder_probe")
    p = meta.extra_params["pruning_strategy"]
    default = p["value"] if isinstance(p, dict) else p
    assert default == "mmmu_encoder_probe"


def test_mmmu_probe_has_pair_fraction_knob():
    """MMMU-specific knob for paired-resolution probing."""
    meta, _ = _get_meta_and_cls("mmmu_encoder_probe")
    assert "probe_pair_fraction" in meta.extra_params
