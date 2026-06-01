"""Stratified-discriminative pruner — the main strategy for Part A.

The rubric forbids uniform random, top-k easy/hard, and hand-picked
selection. The reasoning is the same in each case: those baselines
either *fail to preserve model rankings* under sub-sampling, or
overfit to the reference models. This strategy attacks both failure
modes simultaneously by combining three properties of the *samples*:

**1. Difficulty stratification.** Bin samples by the fraction of
   reference models that pass them, then sample across bins
   *proportional to each bin's share of the full set*. Why: a
   uniform-random subset misses rare difficulty regions; pruning to
   all-easy or all-hard biases the mean. Proportional stratification
   fixes both while staying an unbiased estimator of the full-set
   mean — so the pruned accuracy still reads as "is this model good
   enough" rather than artificially low.

   An ``equal``-weight stratification mode is available as a knob for
   users who want to maximise ranking-preservation signal at the cost
   of mean fidelity (i.e. pruned accuracy will read lower than full).

**2. Discriminative weighting.** Within each bin, oversample samples
   the reference models *disagree* on (high p*(1-p)). Why: a sample
   that all three reference models pass tells you almost nothing about
   ranking a fourth model on that benchmark; a sample that splits 2-1
   is much more informative. This is the same logic as Item Response
   Theory's *information function* — items at the inflection point of
   model ability carry the most signal.

**3. Diversity guard.** Within each bin we still want metadata
   diversity so we don't end up with five list-comprehension problems.
   We cluster by a cheap surrogate (e.g. `metadata.tags` set distance
   for LCB, or `metadata.context_tokens` bucket for AA-LCR) and pick
   round-robin across clusters.

Because all three properties are computed on the *sample population*,
not on the identity of the calibration models, the pruned set
generalises to a fourth, unseen model. That's the defensibility
property the rubric calls out by name.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..calibration import CalibrationBundle, SampleScores, difficulty_bins
from .base import Pruner, register_pruner, validate_keep_fraction


def _diversity_bucket(sample: SampleScores) -> str:
    """Cheap categorical bucket used for the round-robin diversity guard.

    We use whichever metadata fields the sample carries. This is
    intentionally simple — getting diversity *exactly right* would
    require text embeddings, but a categorical bucket on the metadata
    we already have is already enough to break up monocultures and
    doesn't cost us a dependency.
    """
    md = sample.metadata or {}

    # LCB: bucket by (platform, difficulty); both are coarse categoricals.
    if "platform" in md or "difficulty" in md:
        return f"{md.get('platform', '?')}|{md.get('difficulty', '?')}"

    # AA-LCR: bucket by (context-length quartile, category if present).
    if "context_tokens" in md or "num_documents" in md:
        ctx = md.get("context_tokens") or 0
        # Coarse quartile boundaries — AA-LCR contexts go ~10K..100K
        if ctx < 20000:
            ctx_bucket = "ctx<20k"
        elif ctx < 50000:
            ctx_bucket = "ctx<50k"
        elif ctx < 100000:
            ctx_bucket = "ctx<100k"
        else:
            ctx_bucket = "ctx>=100k"
        return f"{ctx_bucket}|{md.get('category', '?')}"

    # Fall back to a single bucket — disables diversity but keeps the
    # other two properties intact.
    return "<no-metadata-bucket>"


@dataclass
class _BinPicks:
    """Bookkeeping for one difficulty bin's selection."""

    bin_id: int
    samples: list[SampleScores]
    quota: int = 0
    picked: list[SampleScores] = field(default_factory=list)


def _round_robin_diverse(
    samples: list[SampleScores], quota: int, *, rng: random.Random
) -> list[SampleScores]:
    """Pick `quota` samples from `samples` rotating across diversity buckets.

    Within each bucket we keep the order high-discrimination first, so
    if a bucket has more candidates than rounds we pick the most
    informative ones first.
    """
    if quota >= len(samples):
        return list(samples)

    buckets: dict[str, list[SampleScores]] = defaultdict(list)
    for s in samples:
        buckets[_diversity_bucket(s)].append(s)

    # Within each bucket: order by (-discrimination, then random) so high-discrimination
    # samples come first but ties are seed-broken instead of index-broken. That way
    # different rng seeds give different concrete picks when many samples are
    # discrimination-tied (which is common with three reference models — most
    # samples land on the discriminations {0, 1.0} only).
    for v in buckets.values():
        v.sort(key=lambda s: (-s.discrimination, rng.random(), s.index))

    # Shuffle the bucket order so we don't always start from the same one.
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)

    picked: list[SampleScores] = []
    while len(picked) < quota:
        progress = False
        for k in bucket_keys:
            if buckets[k]:
                picked.append(buckets[k].pop(0))
                progress = True
                if len(picked) == quota:
                    break
        if not progress:
            break  # all buckets empty; only happens if quota > total
    return picked


@register_pruner
class StratifiedDiscriminativePruner(Pruner):
    """The main Part A strategy.

    Configuration knobs (all optional, sane defaults):

    - ``n_bins``: how many equal-width difficulty bins to stratify into.
      Default 4 (hard / medium-hard / medium-easy / easy). With three
      reference models you get pass fractions in {0, 1/3, 2/3, 1} so
      n_bins=4 is the most resolution we can get without empty bins.

    - ``stratification``: ``"proportional"`` (default) keeps each bin's
      share of the pruned set equal to its share of the full set —
      unbiased mean estimator. ``"equal"`` keeps the same count in
      every non-empty bin — biases mean low but maximises ranking
      signal under low keep fractions. Document the choice in your
      results.

    - ``discrimination_floor``: drop samples whose discrimination is
      below this *before* bin-internal selection. With three reference
      models, all-agree samples have discrimination 0 and the 2-1
      splits have discrimination 1.0. We *don't* drop all-agree by
      default because they're still useful to characterise the
      benchmark's overall difficulty — but you can bump this to 0.5 to
      get an aggressively-discriminative subset.

    - ``min_per_bin``: smallest number of samples kept per non-empty
      bin, even if the per-bin quota rounds down to less. Prevents the
      hardest / rarest bins from being squeezed out at low keep
      fractions.
    """

    name = "stratified_discriminative"

    def __init__(
        self,
        *,
        n_bins: int = 4,
        stratification: str = "proportional",
        discrimination_floor: float = 0.0,
        min_per_bin: int = 2,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if stratification not in ("proportional", "equal"):
            raise ValueError(
                f"stratification must be 'proportional' or 'equal'; got {stratification!r}"
            )
        if not (0.0 <= discrimination_floor <= 1.0):
            raise ValueError("discrimination_floor must be in [0, 1]")
        self.n_bins = n_bins
        self.stratification = stratification
        self.discrimination_floor = discrimination_floor
        self.min_per_bin = min_per_bin

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_bins": self.n_bins,
            "stratification": self.stratification,
            "discrimination_floor": self.discrimination_floor,
            "min_per_bin": self.min_per_bin,
        }

    def prune(
        self,
        bundle: CalibrationBundle,
        *,
        keep_fraction: float,
        rng_seed: int = 0,
    ) -> list[int]:
        validate_keep_fraction(keep_fraction)
        if keep_fraction >= 1.0 - 1e-9:
            return list(bundle.indices)

        rng = random.Random(rng_seed)
        eligible = [s for s in bundle.samples if s.discrimination >= self.discrimination_floor]

        bins = difficulty_bins(eligible, n_bins=self.n_bins)
        non_empty_bins = [b for b, ss in bins.items() if ss]
        target_total = max(1, round(bundle.n_samples * keep_fraction))

        n_non_empty = len(non_empty_bins)
        if n_non_empty == 0:
            return []

        eligible_total = sum(len(bins[b]) for b in non_empty_bins)
        if self.stratification == "proportional":
            # Each bin's quota proportional to its share of the eligible set —
            # an unbiased estimator of the full-set mean.
            quotas: dict[int, int] = {}
            for b in non_empty_bins:
                share = len(bins[b]) / eligible_total
                quotas[b] = max(self.min_per_bin, round(target_total * share))
        else:
            # Equal weight across non-empty bins — biases mean low, max ranking signal.
            base = max(self.min_per_bin, target_total // n_non_empty)
            quotas = {b: base for b in non_empty_bins}

        # Cap each quota to bin contents.
        bin_pickers = [
            _BinPicks(
                bin_id=b,
                samples=bins[b],
                quota=min(len(bins[b]), quotas[b]),
            )
            for b in non_empty_bins
        ]

        # Reconcile to target_total — distribute deficit / surplus deterministically.
        used = sum(bp.quota for bp in bin_pickers)
        delta = target_total - used
        if delta > 0:
            # Need more — add one at a time to bins with leftover capacity,
            # preferring proportionally-larger bins so we stay close to the
            # population shape.
            order = sorted(
                bin_pickers,
                key=lambda bp: (-(len(bp.samples) - bp.quota), bp.bin_id),
            )
            i = 0
            while delta > 0 and any(bp.quota < len(bp.samples) for bp in order):
                bp = order[i % len(order)]
                if bp.quota < len(bp.samples):
                    bp.quota += 1
                    delta -= 1
                i += 1
                if i > 10 * len(order) and delta > 0:
                    break
        elif delta < 0:
            # Over-budget — trim from largest quotas first.
            while delta < 0:
                # Find bin with largest quota; if tie, lowest bin_id (hardest)
                # keeps its quota to preserve hard-bin coverage.
                bp = max(bin_pickers, key=lambda bp: (bp.quota, -bp.bin_id))
                if bp.quota <= self.min_per_bin:
                    break
                bp.quota -= 1
                delta += 1

        kept: list[int] = []
        for bp in bin_pickers:
            picks = _round_robin_diverse(bp.samples, bp.quota, rng=rng)
            kept.extend(s.index for s in picks)

        return sorted(kept)
