"""Load the slim per-sample calibration data into a normalised form.

The pruners want a *per-sample* view: for each `index` (the upstream
sample id), what fraction of reference models passed it, how much they
disagreed, and what metadata is attached. That table is small enough to
live in memory for any benchmark we care about (LCB v5 = 315 rows,
AA-LCR = 100 rows).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default location: the slim copy vendored inside this package.
DEFAULT_CALIBRATION_ROOT = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class SampleScores:
    """One row = one upstream benchmark sample's full calibration record."""

    index: int
    scores: dict[str, float]  # model_name -> score (0/1)
    metadata: dict[str, object]  # union of per-model metadata; per-key last write wins

    @property
    def n_models(self) -> int:
        return len(self.scores)

    @property
    def pass_fraction(self) -> float:
        if not self.scores:
            return float("nan")
        return sum(self.scores.values()) / len(self.scores)

    @property
    def discrimination(self) -> float:
        """How much do the reference models disagree on this sample?

        For binary scores across N models, the variance is maximised at
        p=0.5 (half pass, half fail). We return p*(1-p) normalized to
        [0, 1]: 0 means all models agree (sample tells us nothing about
        ranking), 1 means the models split evenly (maximally
        discriminative).
        """
        p = self.pass_fraction
        if math.isnan(p):
            return 0.0
        # Max possible variance is 0.25 at p=0.5; rescale to [0, 1].
        return 4.0 * p * (1.0 - p)


@dataclass
class CalibrationBundle:
    """All calibration rows for one benchmark."""

    benchmark: str
    samples: list[SampleScores] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure deterministic ordering by upstream index so reruns are stable.
        self.samples.sort(key=lambda s: s.index)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def models(self) -> list[str]:
        return sorted({m for s in self.samples for m in s.scores})

    @property
    def indices(self) -> list[int]:
        return [s.index for s in self.samples]

    def per_model_mean(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for s in self.samples:
            for m, sc in s.scores.items():
                out.setdefault(m, []).append(sc)
        return {m: statistics.fmean(v) for m, v in out.items()}

    def by_index(self) -> dict[int, SampleScores]:
        return {s.index: s for s in self.samples}


def _load_one_model(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"calibration jsonl not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_calibration(benchmark: str, *, root: Path | str | None = None) -> CalibrationBundle:
    """Load all per-model calibration rows for `benchmark` and join on index.

    ``benchmark`` is one of the directories under ``data/`` —
    ``live_code_bench_v5`` or ``aa_lcr`` (MMMU is loaded separately
    because we only have one reference model).
    """
    base = Path(root) if root else DEFAULT_CALIBRATION_ROOT
    bench_dir = base / benchmark
    if not bench_dir.exists():
        raise FileNotFoundError(
            f"calibration directory for {benchmark!r} not found under {base}"
        )

    per_index_scores: dict[int, dict[str, float]] = {}
    per_index_metadata: dict[int, dict[str, object]] = {}

    for fp in sorted(bench_dir.glob("*.jsonl")):
        model = fp.stem
        for row in _load_one_model(fp):
            idx = int(row["index"])
            per_index_scores.setdefault(idx, {})[model] = float(row["score"])
            # Merge metadata across models — they all describe the same sample
            md = row.get("metadata") or {}
            merged = per_index_metadata.setdefault(idx, {})
            for k, v in md.items():
                if v is not None and k not in merged:
                    merged[k] = v
            # AA-LCR carries prediction_len per (sample, model)
            if "prediction_len" in row:
                pl = merged.setdefault("prediction_lens", {})
                pl[model] = int(row["prediction_len"])

    samples = [
        SampleScores(index=idx, scores=scores, metadata=per_index_metadata.get(idx, {}))
        for idx, scores in per_index_scores.items()
    ]
    return CalibrationBundle(benchmark=benchmark, samples=samples)


def difficulty_bins(
    samples: Iterable[SampleScores],
    *,
    n_bins: int = 4,
) -> dict[int, list[SampleScores]]:
    """Group samples into ``n_bins`` equal-width difficulty bins.

    Bin 0 = hardest (lowest pass fraction), bin ``n_bins - 1`` = easiest.
    Empty bins are still returned so callers can detect them and adapt.
    """
    if n_bins < 2:
        raise ValueError("need at least 2 bins to stratify")
    out: dict[int, list[SampleScores]] = {i: [] for i in range(n_bins)}
    for s in samples:
        p = s.pass_fraction
        if math.isnan(p):
            continue
        # bin index: floor(p * n_bins), clamped so p=1.0 lands in last bin
        b = min(int(p * n_bins), n_bins - 1)
        out[b].append(s)
    return out
