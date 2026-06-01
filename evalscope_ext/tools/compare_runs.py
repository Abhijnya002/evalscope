"""``python -m evalscope_ext.tools.compare_runs`` — divergence report.

Implements the rubric's run-contract third line:

::

    python -m evalscope_ext.tools.compare_runs \\
        --full ./results_full/ \\
        --pruned ./results_pruned/

Reads two evalscope output directories (one full run, one pruned run)
and reports how much the pruning shifted the answer:

- Per-model accuracy on the full vs pruned set.
- Spearman rank correlation of the per-model means.
- Top-K retention (K=1, 2, 3) — did the pruned set still pick the
  same winner?
- Maximum absolute mean shift.
- A PASS / FAIL header against the default bar (rho >= 0.7 AND
  mean_shift <= 0.05). A FAIL doesn't mean the pruning is broken —
  it means the *current keep_fraction is too aggressive* for the
  current benchmark + judge; the report shows where the budget has
  to land.

The tool reads evalscope's standard predictions/<model>/<benchmark>.jsonl
output layout. If the run was a pruned run that we know about
(matching the calibration we have), the report also lists the
calibration-side baseline we expected.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from ..pruners.metrics import (
    PrunerQualityReport,
    _spearman,
    _top_k_retention,
)


@dataclass
class _RunSummary:
    """One run's headline numbers."""

    root: Path
    per_model_acc: dict[str, float]
    per_model_n: dict[str, int]


def _load_run(root: Path) -> _RunSummary:
    """Load an evalscope output dir and compute per-model accuracy.

    Evalscope writes predictions to ``<root>/predictions/<model>/<benchmark>_<subset>.jsonl``
    and per-sample reviews to ``<root>/reviews/<model>/<benchmark>_<subset>.jsonl``.
    We compute accuracy from the reviews if present, else fall back to
    the predictions (which may carry an inline score field).
    """
    if not root.exists():
        raise FileNotFoundError(f"run directory not found: {root}")

    per_model_scores: dict[str, list[float]] = {}

    review_root = root / "reviews"
    if review_root.exists():
        for model_dir in review_root.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for fp in model_dir.glob("*.jsonl"):
                for line in fp.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    score = _extract_score(rec)
                    if score is not None:
                        per_model_scores.setdefault(model, []).append(score)

    # Fallback: predictions might carry an inline score (some adapters do)
    if not per_model_scores:
        pred_root = root / "predictions"
        if pred_root.exists():
            for model_dir in pred_root.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                for fp in model_dir.glob("*.jsonl"):
                    for line in fp.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        score = _extract_score(rec)
                        if score is not None:
                            per_model_scores.setdefault(model, []).append(score)

    if not per_model_scores:
        raise ValueError(
            f"no per-sample scores found under {root}. "
            "Expected reviews/<model>/<benchmark>.jsonl or predictions/<model>/<benchmark>.jsonl."
        )

    per_model_acc = {m: statistics.fmean(v) for m, v in per_model_scores.items()}
    per_model_n = {m: len(v) for m, v in per_model_scores.items()}
    return _RunSummary(root=root, per_model_acc=per_model_acc, per_model_n=per_model_n)


def _extract_score(rec: dict) -> float | None:
    """Pull a per-sample score from a record in either reviews or predictions
    format. Supports the evalscope shape (sample_score.score.value.acc / .pass)
    and the simpler {score: float} shape used in unit tests."""
    if "sample_score" in rec:
        sc = rec["sample_score"]
        if isinstance(sc, dict):
            inner = sc.get("score", {})
            if isinstance(inner, dict):
                v = inner.get("value", {})
                if isinstance(v, dict):
                    for k in ("acc", "pass", "accuracy"):
                        if k in v:
                            return float(v[k])
    if "score" in rec and isinstance(rec["score"], (int, float)):
        return float(rec["score"])
    return None


def compare(full: Path, pruned: Path) -> tuple[_RunSummary, _RunSummary, PrunerQualityReport]:
    """Build the divergence report for two run dirs."""
    full_sum = _load_run(full)
    pruned_sum = _load_run(pruned)

    rho = _spearman(full_sum.per_model_acc, pruned_sum.per_model_acc)
    retention = {
        k: _top_k_retention(full_sum.per_model_acc, pruned_sum.per_model_acc, k)
        for k in (1, 2, 3)
    }
    max_shift = max(
        abs(full_sum.per_model_acc[m] - pruned_sum.per_model_acc.get(m, full_sum.per_model_acc[m]))
        for m in full_sum.per_model_acc
    )
    # Reuse the calibration-side report dataclass for a uniform return type.
    report = PrunerQualityReport(
        full_size=sum(full_sum.per_model_n.values()) // max(1, len(full_sum.per_model_n)),
        pruned_size=sum(pruned_sum.per_model_n.values()) // max(1, len(pruned_sum.per_model_n)),
        keep_fraction_observed=(
            (sum(pruned_sum.per_model_n.values()) / sum(full_sum.per_model_n.values()))
            if sum(full_sum.per_model_n.values()) > 0
            else float("nan")
        ),
        per_model_mean_full=full_sum.per_model_acc,
        per_model_mean_pruned=pruned_sum.per_model_acc,
        spearman_rank_correlation=rho,
        top_k_retention=retention,
        max_abs_mean_shift=max_shift,
    )
    return full_sum, pruned_sum, report


def format_report(
    full_sum: _RunSummary, pruned_sum: _RunSummary, report: PrunerQualityReport
) -> str:
    lines = []
    bar = "PASS" if report.passes_default_bar else "FAIL"
    lines.append("=" * 70)
    lines.append(f" Pruning divergence report — {bar}")
    lines.append("=" * 70)
    lines.append(f"  Full set    : {full_sum.root}")
    lines.append(f"                {report.full_size} samples per model")
    lines.append(f"  Pruned set  : {pruned_sum.root}")
    lines.append(
        f"                {report.pruned_size} samples per model "
        f"(observed keep fraction = {report.keep_fraction_observed:.2%})"
    )
    lines.append("")
    lines.append("  Per-model accuracy:")
    lines.append(f"    {'model':<25} {'full':>10} {'pruned':>10} {'shift':>10}")
    for m in sorted(report.per_model_mean_full):
        f_acc = report.per_model_mean_full[m]
        p_acc = report.per_model_mean_pruned.get(m, float("nan"))
        shift = p_acc - f_acc if p_acc == p_acc else float("nan")
        lines.append(f"    {m:<25} {f_acc:>10.4f} {p_acc:>10.4f} {shift:>+10.4f}")
    lines.append("")
    lines.append(f"  Spearman rank-correlation : {report.spearman_rank_correlation:+.4f}  (>= 0.7 to PASS)")
    lines.append(f"  Max abs mean shift        : {report.max_abs_mean_shift:.4f}  (<= 0.05 to PASS)")
    lines.append("  Top-K retention:")
    for k, v in sorted(report.top_k_retention.items()):
        lines.append(f"    K={k}: {v:.2%}")
    lines.append("")
    if not report.passes_default_bar:
        lines.append("  ! FAIL means the current keep_fraction is too aggressive for this")
        lines.append("    (benchmark, model set). Either widen prune_ratio or — for AA-LCR —")
        lines.append("    switch pruning_strategy to judge_noise_aware. See README.")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an evalscope full-set run to a pruned-set run."
    )
    parser.add_argument("--full", type=Path, required=True, help="Full-run output dir")
    parser.add_argument("--pruned", type=Path, required=True, help="Pruned-run output dir")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of the formatted table.",
    )
    args = parser.parse_args(argv)

    try:
        full_sum, pruned_sum, report = compare(args.full, args.pruned)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "full_dir": str(full_sum.root),
                    "pruned_dir": str(pruned_sum.root),
                    "per_model_full": report.per_model_mean_full,
                    "per_model_pruned": report.per_model_mean_pruned,
                    "spearman_rank_correlation": report.spearman_rank_correlation,
                    "max_abs_mean_shift": report.max_abs_mean_shift,
                    "top_k_retention": report.top_k_retention,
                    "passes_default_bar": report.passes_default_bar,
                },
                indent=2,
            )
        )
    else:
        print(format_report(full_sum, pruned_sum, report))
    return 0 if report.passes_default_bar else 1


if __name__ == "__main__":
    raise SystemExit(main())
