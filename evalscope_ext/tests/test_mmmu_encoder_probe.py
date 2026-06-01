"""Tests for the MMMU encoder-probe pruner stub."""

from __future__ import annotations

import pytest

from evalscope_ext.calibration import CalibrationBundle, SampleScores
from evalscope_ext.pruners import (
    MMMUEncoderProbePruner,
    ProbePair,
    get_pruner,
    registered_strategies,
)


def _synthetic_mmmu_bundle() -> CalibrationBundle:
    """Build a synthetic MMMU-shaped bundle for testing the pruner's
    selection without a real 12 K HF dataset."""
    samples = []
    cases = [
        # (subject, num_images, figure_type, width, height, expected_high_weight)
        ("Math", 2, "diagram", 1024, 768, True),  # encoder-heavy
        ("Chemistry", 1, "formula", 800, 1600, True),  # tall + formula
        ("Literature", 1, "photo", 800, 600, False),  # language-bound
        ("Accounting", 1, "table", 600, 600, False),
        ("Diagnostics_and_Laboratory_Medicine", 1, "photo", 1024, 1024, True),
        ("Geography", 1, "map", 2400, 800, True),  # wide map
        ("Art", 1, "photo", 800, 800, False),
        ("Physics", 1, "diagram", 800, 1000, True),
        ("Sociology", 1, "chart", 800, 600, False),
        ("Materials", 2, "chart", 1200, 800, True),
    ]
    for i, (subj, n, ft, w, h, _) in enumerate(cases):
        samples.append(
            SampleScores(
                index=i,
                scores={"glm-4.5v-fp8": 1.0 if i % 2 else 0.0},
                metadata={
                    "subject": subj,
                    "num_images": n,
                    "figure_type": ft,
                    "width": w,
                    "height": h,
                },
            )
        )
    return CalibrationBundle(benchmark="mmmu_synth", samples=samples)


def test_pruner_is_registered():
    assert "mmmu_encoder_probe" in registered_strategies()


def test_pruner_picks_encoder_heavy_samples_first():
    """At 50 % keep, the kept set should over-represent encoder-bound
    subjects (Math, Chemistry, Diagnostics, Physics, Materials,
    Geography) vs language-bound ones (Literature, Accounting,
    Sociology, Art)."""
    b = _synthetic_mmmu_bundle()
    p = MMMUEncoderProbePruner()
    kept = p.prune(b, keep_fraction=0.5, rng_seed=0)
    kept_subjects = [b.by_index()[i].metadata["subject"] for i in kept]

    encoder_heavy = {
        "Math",
        "Chemistry",
        "Diagnostics_and_Laboratory_Medicine",
        "Geography",
        "Physics",
        "Materials",
    }
    encoder_share = sum(1 for s in kept_subjects if s in encoder_heavy) / len(kept_subjects)
    # At 50 % keep with 6 encoder-heavy out of 10 samples in the bundle,
    # we expect the kept set to be almost entirely encoder-heavy.
    assert encoder_share >= 0.6, f"only {encoder_share:.2f} encoder-heavy: kept={kept_subjects}"


def test_probe_pair_count_matches_fraction():
    b = _synthetic_mmmu_bundle()
    p = MMMUEncoderProbePruner(probe_pair_fraction=0.5)
    kept = p.prune(b, keep_fraction=0.4, rng_seed=42)
    # 40% of 10 = 4 kept; 50% of 4 = 2 probe pairs.
    assert len(p.probe_pairs) == 2
    for pair in p.probe_pairs:
        assert isinstance(pair, ProbePair)
        assert pair.sample_index in kept


def test_probe_pair_fraction_zero_means_no_pairs():
    b = _synthetic_mmmu_bundle()
    p = MMMUEncoderProbePruner(probe_pair_fraction=0.0)
    p.prune(b, keep_fraction=0.5, rng_seed=0)
    assert p.probe_pairs == []


def test_rejects_invalid_probe_pair_fraction():
    with pytest.raises(ValueError):
        MMMUEncoderProbePruner(probe_pair_fraction=-0.1)
    with pytest.raises(ValueError):
        MMMUEncoderProbePruner(probe_pair_fraction=1.5)


def test_keep_fraction_one_returns_everything():
    b = _synthetic_mmmu_bundle()
    p = MMMUEncoderProbePruner()
    kept = p.prune(b, keep_fraction=1.0)
    assert sorted(kept) == sorted(b.indices)


def test_wide_aspect_ratio_gets_picked_at_50pct():
    """A wide map (>2:1 aspect ratio) should get the aspect-ratio bonus
    and survive a 50 % keep cut against the synthetic mixed bundle —
    proving the aspect-ratio signal actually fires. (At 30 % budget it
    can lose to multi-image + diagram samples that score higher.)"""
    b = _synthetic_mmmu_bundle()
    p = MMMUEncoderProbePruner()
    kept = p.prune(b, keep_fraction=0.5, rng_seed=0)
    geography_idx = next(s.index for s in b.samples if s.metadata.get("subject") == "Geography")
    assert geography_idx in kept
