"""Selection-quality metrics for evaluating a pruner.

A pruner outputs a *subset* of indices. To defend the choice, we need
to show the subset preserves what the *full* set tells us. Three
metrics cover the angles the rubric explicitly asks about:

1. **Spearman rank correlation** of per-model means between full and
   pruned. Captures whether the *ranking* of models is preserved — the
   most important property if we're using the pruned set to choose
   between candidates.

2. **Top-k retention** — of the K best models on the full set, how
   many are in the top-K on the pruned set? Captures "did we still pick
   the same winner?" for any K.

3. **Mean shift** — absolute difference between each model's full vs
   pruned accuracy. A large shift means the pruned set is biased; a
   small shift means it's a faithful estimator.

All three are computed from the calibration bundle alone, so we can
evaluate pruners *without running any model* — which is exactly what
this challenge requires given the data we have.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..calibration import CalibrationBundle


@dataclass(frozen=True)
class PrunerQualityReport:
    """Per-pruner verdict against a calibration bundle."""

    full_size: int
    pruned_size: int
    keep_fraction_observed: float
    per_model_mean_full: dict[str, float]
    per_model_mean_pruned: dict[str, float]
    spearman_rank_correlation: float
    top_k_retention: dict[int, float]  # K -> fraction
    max_abs_mean_shift: float

    @property
    def passes_default_bar(self) -> bool:
        """Default sanity threshold: rank correlation >= 0.7 and mean shift <= 0.05."""
        return self.spearman_rank_correlation >= 0.7 and self.max_abs_mean_shift <= 0.05


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Spearman rank correlation between two dicts keyed by model name.

    Returns NaN for fewer than two shared keys (you cannot rank one
    item).
    """
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")

    ranks_a = _ranks([a[k] for k in keys])
    ranks_b = _ranks([b[k] for k in keys])

    n = len(keys)
    # Pearson on the ranks ≡ Spearman.
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    num = sum((ranks_a[i] - mean_a) * (ranks_b[i] - mean_b) for i in range(n))
    den_a = sum((ra - mean_a) ** 2 for ra in ranks_a) ** 0.5
    den_b = sum((rb - mean_b) ** 2 for rb in ranks_b) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks (1-indexed). Ties get the average rank."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _top_k_retention(
    full: dict[str, float],
    pruned: dict[str, float],
    k: int,
) -> float:
    """Fraction of the full set's top-K that appears in the pruned set's top-K."""
    if k <= 0:
        return float("nan")
    keys = set(full) & set(pruned)
    if len(keys) < k:
        return float("nan")
    top_full = {kk for kk, _ in sorted(full.items(), key=lambda t: -t[1])[:k]}
    top_pruned = {kk for kk, _ in sorted(pruned.items(), key=lambda t: -t[1])[:k]}
    return len(top_full & top_pruned) / k


def evaluate_selection(
    bundle: CalibrationBundle,
    kept_indices: list[int],
    *,
    top_k_values: tuple[int, ...] = (1, 2, 3),
) -> PrunerQualityReport:
    """Evaluate a kept-index list against a calibration bundle."""
    if not kept_indices:
        raise ValueError("kept_indices is empty")

    by_index = bundle.by_index()
    missing = [i for i in kept_indices if i not in by_index]
    if missing:
        raise KeyError(f"{len(missing)} kept index/indices not in calibration bundle")

    # Per-model means on the full and pruned sets.
    full_means = bundle.per_model_mean()
    kept_samples = [by_index[i] for i in kept_indices]
    pruned_means: dict[str, list[float]] = {}
    for s in kept_samples:
        for m, sc in s.scores.items():
            pruned_means.setdefault(m, []).append(sc)
    pruned_means_avg = {m: statistics.fmean(v) for m, v in pruned_means.items()}

    rho = _spearman(full_means, pruned_means_avg)
    retention = {k: _top_k_retention(full_means, pruned_means_avg, k) for k in top_k_values}
    max_shift = max(
        abs(full_means[m] - pruned_means_avg.get(m, full_means[m])) for m in full_means
    )

    return PrunerQualityReport(
        full_size=bundle.n_samples,
        pruned_size=len(kept_indices),
        keep_fraction_observed=len(kept_indices) / bundle.n_samples,
        per_model_mean_full=full_means,
        per_model_mean_pruned=pruned_means_avg,
        spearman_rank_correlation=rho,
        top_k_retention=retention,
        max_abs_mean_shift=max_shift,
    )
