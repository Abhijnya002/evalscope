"""``aa_lcr_pruned`` — pruned variant of AA-LCR.

Wraps the upstream ``aa_lcr`` adapter with an index-keyed sample
filter driven by a pruner. Same run-contract shape as
``live_code_bench_pruned``::

    evalscope eval --model gpt-oss-120b --datasets aa_lcr_pruned \\
      --dataset-args '{"pruning_strategy": "judge_noise_aware",
                       "prune_ratio": 0.3,
                       "rng_seed": 42}' \\
      --output ./results_pruned/

The default ``pruning_strategy`` here is ``judge_noise_aware`` rather
than ``stratified_discriminative``, because AA-LCR is LLM-judged and
the noise penalty is what makes the pruned subset's rank correlation
defendable at 30 % keep (the smaller benchmark needs a higher keep
ratio than LCB does).
"""

from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.constants import Tags

from ..calibration import load_calibration
from ..pruners import get_pruner, registered_strategies, validate_keep_fraction

_EXTRA_PARAMS = {
    "pruning_strategy": {
        "type": "str",
        "description": (
            "One of: " + ", ".join(registered_strategies())
            + ". Defaults to judge_noise_aware (recommended for AA-LCR)."
        ),
        "value": "judge_noise_aware",
    },
    "prune_ratio": {
        "type": "float",
        "description": (
            "Keep this fraction of the full set. AA-LCR is small (100 samples) and"
            " LLM-judged; we recommend prune_ratio >= 0.3 for defensible rank-corr."
        ),
        "value": 0.3,
    },
    "rng_seed": {"type": "int", "value": 0},
    "calibration_benchmark": {
        "type": "str",
        "description": "Calibration directory name. Defaults to aa_lcr.",
        "value": "aa_lcr",
    },
}


@register_benchmark(
    BenchmarkMeta(
        name="aa_lcr_pruned",
        pretty_name="AA-LCR (pruned)",
        tags=[Tags.REASONING, Tags.LONG_CONTEXT] if hasattr(Tags, "LONG_CONTEXT") else [Tags.REASONING],
        description=(
            "Pruned variant of AA-LCR. Default strategy is judge_noise_aware "
            "to compensate for the LLM-judge noise the rubric calls out."
        ),
        dataset_id="evalscope/AA-LCR",
        metric_list=["acc"],
        eval_split="test",
        extra_params=_EXTRA_PARAMS,
    )
)
class AaLcrPrunedAdapter(AALCRAdapter):
    """AA-LCR adapter that emits only the pruner-selected subset of samples."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._strategy_name: str = self.extra_params.get(
            "pruning_strategy", "judge_noise_aware"
        )
        self._prune_ratio: float = float(self.extra_params.get("prune_ratio", 0.3))
        self._rng_seed: int = int(self.extra_params.get("rng_seed", 0))
        self._calibration_benchmark: str = self.extra_params.get(
            "calibration_benchmark", "aa_lcr"
        )
        validate_keep_fraction(self._prune_ratio)

        bundle = load_calibration(self._calibration_benchmark)
        pruner = get_pruner(self._strategy_name)
        self._kept_indices: set[int] = set(
            pruner.prune(
                bundle,
                keep_fraction=self._prune_ratio,
                rng_seed=self._rng_seed,
            )
        )

    def sample_filter(self, sample: Any) -> bool:  # type: ignore[override]
        md = sample.metadata or {}
        idx = md.get("__calibration_index")
        if idx is not None and int(idx) not in self._kept_indices:
            return False
        # AALCRAdapter may not have a sample_filter — guard the super call.
        super_filter = getattr(super(), "sample_filter", None)
        if super_filter is None:
            return True
        return super_filter(sample)

    def record_to_sample(self, record: Dict[str, Any]):  # type: ignore[override]
        sample = super().record_to_sample(record)
        if sample.metadata is None:
            sample.metadata = {}
        # AA-LCR records carry a top-level "index" field in the shipped data;
        # use it if present, else fall back to a monotonic row counter.
        if "index" in record:
            sample.metadata["__calibration_index"] = int(record["index"])
        else:
            sample.metadata.setdefault("__calibration_index", self._row_counter_next())
        return sample

    _row_counter: int = -1

    def _row_counter_next(self) -> int:
        self._row_counter = (self._row_counter or -1) + 1
        return self._row_counter

    def describe_pruning(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy_name,
            "prune_ratio": self._prune_ratio,
            "rng_seed": self._rng_seed,
            "calibration_benchmark": self._calibration_benchmark,
            "n_kept": len(self._kept_indices),
        }
