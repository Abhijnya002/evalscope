"""``aa_lcr_pruned`` - pruned variant of AA-LCR.

Same universal pattern as :mod:`live_code_bench_pruned`. The defaults
differ because AA-LCR is LLM-judged and small (100 samples), so the
recommended strategy is the noise-aware variant at a 30% keep ratio.
"""

from __future__ import annotations

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.constants import Tags

from ._base import PrunedMixin, _shared_extra_params

_TAGS = [Tags.REASONING]
if hasattr(Tags, "LONG_CONTEXT"):
    _TAGS.append(Tags.LONG_CONTEXT)

_EXTRA_PARAMS = {
    **_shared_extra_params("judge_noise_aware", 0.3),
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
        tags=_TAGS,
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
class AaLcrPrunedAdapter(PrunedMixin, AALCRAdapter):
    """AA-LCR adapter that emits only the pruner-selected subset."""

    DEFAULT_STRATEGY = "judge_noise_aware"
    DEFAULT_PRUNE_RATIO = 0.3
    RECORD_INDEX_FIELD = "index"  # AA-LCR raw records carry a top-level "index"

    @property
    def CALIBRATION_BENCHMARK(self) -> str:  # type: ignore[override]
        params = self.extra_params or {}
        return params.get("calibration_benchmark", "aa_lcr")
