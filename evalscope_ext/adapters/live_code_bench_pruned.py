"""``live_code_bench_pruned`` - pruned variant of LiveCodeBench.

Built from the universal ``PrunedMixin`` plus the upstream
:class:`LiveCodeBenchAdapter`. The mixin owns all the
``--dataset-args`` plumbing and the kept-indices filter. This file is
just the class-level config plus the LCB-specific record-index field.
"""

from __future__ import annotations

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
    LiveCodeBenchAdapter,
)
from evalscope.constants import Tags

from ._base import PrunedMixin, _shared_extra_params

_EXTRA_PARAMS = {
    **_shared_extra_params("stratified_discriminative", 0.1),
    "calibration_benchmark": {
        "type": "str",
        "description": (
            "Calibration directory name under evalscope_ext/calibration/data/. "
            "Defaults to live_code_bench_v5."
        ),
        "value": "live_code_bench_v5",
    },
    "start_date": {"type": "str | null", "value": None},
    "end_date": {"type": "str | null", "value": None},
}


@register_benchmark(
    BenchmarkMeta(
        name="live_code_bench_pruned",
        pretty_name="Live-Code-Bench (pruned)",
        tags=[Tags.CODING],
        description=(
            "Pruned variant of LiveCodeBench. Selects ~10-30% of v5 via "
            "the stratified-discriminative pruner. Wraps upstream "
            "live_code_bench. See evalscope_ext/README.md for the policy."
        ),
        dataset_id="evalscope/livecodebench_code_generation_lite_parquet",
        subset_list=["v5"],
        metric_list=["acc"],
        aggregation="mean_and_pass_at_k",
        eval_split="test",
        prompt_template=(
            "### Question:\n{question_content}\n\n"
            "{format_prompt} ### Answer: (use the provided format with backticks)\n\n"
        ),
        review_timeout=6,
        extra_params=_EXTRA_PARAMS,
        sandbox_config={
            "image": "python:3.11-slim",
            "tools_config": {"shell_executor": {}, "python_executor": {}},
        },
    )
)
class LiveCodeBenchPrunedAdapter(PrunedMixin, LiveCodeBenchAdapter):
    """LCB adapter that emits only the pruner-selected subset of samples."""

    DEFAULT_STRATEGY = "stratified_discriminative"
    DEFAULT_PRUNE_RATIO = 0.1
    # The upstream LCB record carries question_id as the stable per-sample
    # identifier inside evaluation_sample. We look it up from the raw
    # record dict so the stamping survives across reruns.
    RECORD_INDEX_FIELD = None  # falls back to row counter; LCB has no top-level id

    @property
    def CALIBRATION_BENCHMARK(self) -> str:  # type: ignore[override]
        # Read from extra_params so a caller can point at a different
        # calibration directory (e.g. a future v6 release).
        params = self.extra_params or {}
        return params.get("calibration_benchmark", "live_code_bench_v5")
