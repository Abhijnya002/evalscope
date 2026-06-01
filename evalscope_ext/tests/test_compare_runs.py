"""Tests for the compare_runs CLI against synthetic evalscope output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalscope_ext.tools.compare_runs import compare, format_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_run(
    root: Path,
    benchmark: str,
    model_scores: dict[str, list[float]],
) -> None:
    """Write an evalscope-shaped output tree under `root`."""
    for model, scores in model_scores.items():
        rows = [
            {
                "index": i,
                "sample_score": {"score": {"value": {"acc": s}}},
            }
            for i, s in enumerate(scores)
        ]
        _write_jsonl(root / "reviews" / model / f"{benchmark}.jsonl", rows)


def test_compare_picks_up_per_model_accuracies(tmp_path):
    full = tmp_path / "full"
    pruned = tmp_path / "pruned"
    _make_run(full, "live_code_bench_v5", {"A": [1.0] * 100, "B": [1.0] * 50 + [0.0] * 50})
    _make_run(pruned, "live_code_bench_v5", {"A": [1.0] * 10, "B": [1.0] * 5 + [0.0] * 5})

    full_sum, pruned_sum, report = compare(full, pruned)
    assert full_sum.per_model_acc["A"] == 1.0
    assert full_sum.per_model_acc["B"] == 0.5
    assert pruned_sum.per_model_acc["A"] == 1.0
    assert pruned_sum.per_model_acc["B"] == 0.5
    import math
    assert math.isclose(report.spearman_rank_correlation, 1.0, abs_tol=1e-9)
    assert report.max_abs_mean_shift == 0.0


def test_compare_detects_rank_inversion(tmp_path):
    """Full set: A > B. Pruned set: B > A. The CLI should catch it."""
    full = tmp_path / "full"
    pruned = tmp_path / "pruned"
    _make_run(full, "aa_lcr", {"A": [1.0] * 80 + [0.0] * 20, "B": [1.0] * 60 + [0.0] * 40})  # A:0.8, B:0.6
    _make_run(pruned, "aa_lcr", {"A": [1.0] * 3 + [0.0] * 7, "B": [1.0] * 7 + [0.0] * 3})  # A:0.3, B:0.7

    _, _, report = compare(full, pruned)
    assert report.spearman_rank_correlation < 0.0  # inverted ranking
    assert not report.passes_default_bar


def test_compare_passes_default_bar_when_pruning_is_faithful(tmp_path):
    full = tmp_path / "full"
    pruned = tmp_path / "pruned"
    # Full set: clear A > B > C ordering. Pruned faithfully preserves it.
    _make_run(
        full,
        "live_code_bench_v5",
        {"A": [1.0] * 80 + [0.0] * 20, "B": [1.0] * 65 + [0.0] * 35, "C": [1.0] * 55 + [0.0] * 45},
    )
    _make_run(
        pruned,
        "live_code_bench_v5",
        {"A": [1.0] * 12 + [0.0] * 3, "B": [1.0] * 10 + [0.0] * 5, "C": [1.0] * 8 + [0.0] * 7},
    )
    _, _, report = compare(full, pruned)
    assert report.passes_default_bar


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare(tmp_path / "no_such_dir", tmp_path / "also_missing")


def test_format_report_smoke(tmp_path):
    full = tmp_path / "full"
    pruned = tmp_path / "pruned"
    _make_run(full, "x", {"A": [1.0] * 10})
    _make_run(pruned, "x", {"A": [1.0] * 5})
    full_sum, pruned_sum, report = compare(full, pruned)
    rendered = format_report(full_sum, pruned_sum, report)
    assert "PASS" in rendered or "FAIL" in rendered
    assert "Per-model accuracy" in rendered
    assert "Spearman" in rendered


def test_cli_main_returns_exit_code_zero_when_passing(tmp_path):
    full = tmp_path / "full"
    pruned = tmp_path / "pruned"
    _make_run(full, "lcb", {"A": [1.0] * 100, "B": [1.0] * 50 + [0.0] * 50})
    _make_run(pruned, "lcb", {"A": [1.0] * 10, "B": [1.0] * 5 + [0.0] * 5})
    proc = subprocess.run(
        [sys.executable, "-m", "evalscope_ext.tools.compare_runs",
         "--full", str(full), "--pruned", str(pruned)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0
    assert "PASS" in proc.stdout


def test_cli_json_mode_emits_valid_json(tmp_path):
    full = tmp_path / "full"
    pruned = tmp_path / "pruned"
    _make_run(full, "lcb", {"A": [1.0] * 100, "B": [1.0] * 50 + [0.0] * 50})
    _make_run(pruned, "lcb", {"A": [1.0] * 10, "B": [1.0] * 5 + [0.0] * 5})
    proc = subprocess.run(
        [sys.executable, "-m", "evalscope_ext.tools.compare_runs",
         "--full", str(full), "--pruned", str(pruned), "--json"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["per_model_full"]["A"] == 1.0
    assert "spearman_rank_correlation" in data
    assert data["passes_default_bar"] is True
