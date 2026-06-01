"""AA-LCR variant — judge-noise-aware pruner.

AA-LCR is graded by an LLM judge. The challenge spec calls this out
explicitly: *"AA-LCR is graded by an LLM judge, which is
non-deterministic. Any variance analysis you do on AA-LCR will partly
measure judge noise, not sample variance."*

That changes the prune problem. The pure stratified-discriminative
pruner treats sample-level disagreement as a signal of *sample
discriminativeness*. But for AA-LCR, some of the disagreement is
*judge noise*: the same model on the same sample could plausibly get
0.0 one run and 1.0 the next. Selecting heavily on those samples
amplifies noise instead of signal.

We don't have repeat-judge runs in the shipped data, so we can't
*directly* estimate per-sample judge variance. But we can use a proxy:
**samples where the three reference models split 2-1 but the
prediction-length distribution is bimodal are more likely to be
genuine difficulty divergence**, whereas samples where the
prediction lengths are similar across all three are more likely to be
judge-noise driven. Same answer, judge calls one right and one wrong.

Combining that with the parent's stratification + discrimination
gives us:

1. Bin by difficulty (same as parent).
2. Within each bin, prefer discriminative samples (same as parent).
3. *Among discriminative samples*, down-weight those that look
   judge-noise driven (new — see :func:`_judge_noise_score`).
4. Diversity round-robin across context-length buckets (same as
   parent).

We also boost ``min_per_bin`` for AA-LCR because the benchmark is so
small (100 samples). Losing the only-3 hardest samples leaves the
pruned set with no signal at all about the hardest region.
"""

from __future__ import annotations

from typing import Any

from ..calibration import CalibrationBundle, SampleScores
from .base import register_pruner
from .stratified import StratifiedDiscriminativePruner


def _judge_noise_score(sample: SampleScores) -> float:
    """Return a noise-likelihood score in [0, 1].

    Higher means *more likely to be judge noise*, lower means *more
    likely to be real disagreement*. Built from cheap signals available
    in the slim calibration data:

    - For samples where one model passes and two fail (or vice versa):
      compute the coefficient of variation (cv) of the three models'
      prediction lengths. If the three models produced similarly-long
      outputs but the judge graded only one as correct, that's
      suspicious — the judge probably made a calibration error.
    - For all-agree samples: noise score 0 (no disagreement to be
      noisy about).
    - For samples we can't compute prediction-length cv on (missing
      metadata): noise score 0.5 (we cannot rule it in or out).
    """
    p = sample.pass_fraction
    # all-agree → no disagreement, no judge noise to score
    if p == 0.0 or p == 1.0:
        return 0.0

    lens = (sample.metadata or {}).get("prediction_lens", {})
    if not lens or len(lens) < 2:
        return 0.5

    values = [v for v in lens.values() if v > 0]
    if not values or len(values) < 2:
        return 0.5

    mean = sum(values) / len(values)
    if mean == 0:
        return 0.5
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    cv = std / mean  # coefficient of variation

    # cv ~ 0  →  similar lengths (suspicious, judge probably noisy)
    # cv > 0.5 →  very different lengths (genuine difficulty divergence)
    # Map cv to a noise score in [0, 1] with a soft threshold at 0.3.
    if cv >= 0.3:
        return 0.0
    return 1.0 - (cv / 0.3)


@register_pruner
class JudgeNoiseAwarePruner(StratifiedDiscriminativePruner):
    """Stratified-discriminative + a judge-noise penalty.

    The shipped AA-LCR data is LLM-judged and small (100 samples). Two
    concrete adjustments vs. the parent class:

    - During the within-bin sort, the effective discrimination of a
      sample is multiplied by ``(1 - judge_noise_score)`` so that
      suspected judge-noise items sink to the bottom of their bin.
    - ``min_per_bin`` defaults to 4 instead of 2 — at 100 samples, the
      hardest bin can have only 5-8 items, and a smaller floor would
      let very low keep_fractions drop it entirely.

    Everything else (proportional stratification, diversity round-robin,
    seed-stability) inherits unchanged from
    :class:`StratifiedDiscriminativePruner`.
    """

    name = "judge_noise_aware"

    def __init__(
        self,
        *,
        n_bins: int = 4,
        stratification: str = "proportional",
        discrimination_floor: float = 0.0,
        min_per_bin: int = 4,
        noise_penalty_weight: float = 1.0,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            stratification=stratification,
            discrimination_floor=discrimination_floor,
            min_per_bin=min_per_bin,
        )
        if not (0.0 <= noise_penalty_weight <= 1.0):
            raise ValueError("noise_penalty_weight must be in [0, 1]")
        self.noise_penalty_weight = noise_penalty_weight

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["noise_penalty_weight"] = self.noise_penalty_weight
        return d

    def prune(
        self,
        bundle: CalibrationBundle,
        *,
        keep_fraction: float,
        rng_seed: int = 0,
    ) -> list[int]:
        # We can't override the parent's bin-internal sort without
        # restructuring it, so we *adjust the underlying SampleScores'
        # discrimination signal* by re-wrapping samples with a
        # noise-penalised view. The simplest place to do that is to
        # pre-modify the bundle's samples — but SampleScores is frozen.
        #
        # Instead we monkey-patch a noise-aware sample list onto a copy
        # of the bundle before delegating to super().prune(). This
        # keeps the policy expression local without duplicating the
        # parent's stratification logic.

        from dataclasses import replace

        adjusted_samples = []
        for s in bundle.samples:
            penalty = _judge_noise_score(s) * self.noise_penalty_weight
            if penalty == 0.0:
                adjusted_samples.append(s)
                continue
            # Build a new SampleScores with adjusted scores so
            # discrimination shrinks proportionally. We can't mutate the
            # frozen dataclass, so we make a new bundle.
            new_md = dict(s.metadata or {})
            new_md["_noise_penalty"] = penalty
            adjusted_samples.append(
                replace(
                    s,
                    metadata=new_md,
                    # Don't actually change scores — that would corrupt downstream
                    # metric calculation. Instead we use the metadata hint and
                    # override the sort key by injecting a derived attribute via
                    # the parent's `_round_robin_diverse` sort path. See note below.
                )
            )

        # We need to *signal* the noise penalty into the parent's sort
        # order. The parent's sort uses (-discrimination, rng.random(),
        # index). We achieve "sink noisy samples" by reducing the
        # effective discrimination through a wrapper.
        from ..calibration import CalibrationBundle as _Bundle, SampleScores as _S

        class _NoisyAwareSampleScores(_S):
            __slots__ = ()

            @property
            def discrimination(self) -> float:  # type: ignore[override]
                base = super().discrimination
                penalty = (self.metadata or {}).get("_noise_penalty", 0.0)
                return base * (1.0 - penalty)

        new_samples = [
            _NoisyAwareSampleScores(
                index=s.index, scores=s.scores, metadata=s.metadata or {}
            )
            for s in adjusted_samples
        ]
        adjusted_bundle = _Bundle(benchmark=bundle.benchmark, samples=new_samples)

        return super().prune(
            adjusted_bundle, keep_fraction=keep_fraction, rng_seed=rng_seed
        )
