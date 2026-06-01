"""``live_code_bench_pruned`` — pruned variant of LiveCodeBench.

Wraps the upstream :class:`LiveCodeBenchAdapter` with a sample-filter
that drops samples whose dataset index isn't in the pruner's output.
The strategy + ratio are taken from the rubric's run-contract
``--dataset-args`` JSON:

::

    evalscope eval --model gpt-oss-120b --datasets live_code_bench_pruned \\
      --dataset-args '{"pruning_strategy": "stratified_discriminative",
                       "prune_ratio": 0.1,
                       "rng_seed": 42}' \\
      --output ./results_pruned/

Implementation note: the upstream adapter uses ``sample_filter`` to
drop samples by ``contest_date``. We use the same hook but key off the
sample's enumeration position, matched against the calibration data's
``index`` field. The shipped Cerebras ``Evals/`` predictions use the
same ``index`` keying, so the pruner picks indices straight from the
calibration bundle.
"""

from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
    LiveCodeBenchAdapter,
)
from evalscope.constants import Tags

from ..calibration import load_calibration
from ..pruners import get_pruner, registered_strategies, validate_keep_fraction

# Build the run-contract spec for ``--dataset-args``. We re-export only
# the parameters that matter for pruning; the upstream adapter's date /
# debug parameters still work via the underlying class because we
# inherit unchanged.
_EXTRA_PARAMS = {
    "pruning_strategy": {
        "type": "str",
        "description": (
            "One of: " + ", ".join(registered_strategies())
            + ". Defaults to stratified_discriminative."
        ),
        "value": "stratified_discriminative",
    },
    "prune_ratio": {
        "type": "float",
        "description": "Keep this fraction of the full set. Must be in (0, 1].",
        "value": 0.1,
    },
    "rng_seed": {
        "type": "int",
        "description": "Seed for the rng used by the pruner.",
        "value": 0,
    },
    "calibration_benchmark": {
        "type": "str",
        "description": (
            "Calibration directory name under evalscope_ext/calibration/data/. "
            "Defaults to live_code_bench_v5."
        ),
        "value": "live_code_bench_v5",
    },
    # Keep the parent's date filters available so a caller can compose
    # them with pruning if they want a pruned + date-filtered run.
    "start_date": {"type": "str | null", "value": None},
    "end_date": {"type": "str | null", "value": None},
}


@register_benchmark(
    BenchmarkMeta(
        name="live_code_bench_pruned",
        pretty_name="Live-Code-Bench (pruned)",
        tags=[Tags.CODING],
        description=(
            "Pruned variant of LiveCodeBench. Selects ~10–30 % of v5 "
            "via the stratified-discriminative pruner (or any registered "
            "strategy). See evalscope_ext/README.md for the policy."
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
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """LCB adapter that emits only the pruner-selected subset of samples."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Read pruning config from --dataset-args (plumbed via extra_params)
        self._strategy_name: str = self.extra_params.get(
            "pruning_strategy", "stratified_discriminative"
        )
        self._prune_ratio: float = float(self.extra_params.get("prune_ratio", 0.1))
        self._rng_seed: int = int(self.extra_params.get("rng_seed", 0))
        self._calibration_benchmark: str = self.extra_params.get(
            "calibration_benchmark", "live_code_bench_v5"
        )

        validate_keep_fraction(self._prune_ratio)

        # Resolve the kept-indices once at construction; this is cheap (~200 KB read
        # + a stratified sort), and it means every sample_filter() call is O(1).
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
        """Keep samples whose calibration index is in the pruner's output.

        Composed with the parent's date filter via the underlying
        ``LiveCodeBenchAdapter.sample_filter`` — we apply the index filter
        first and short-circuit if it drops the sample, then defer to the
        parent for any date filtering the caller asked for.
        """
        # Upstream LCB carries the sample's index inside ``metadata`` via
        # the ``evaluation_sample`` field; the calibration jsonl indices
        # match that field's per-platform problem ID. The parent adapter
        # uses ``contest_date`` for date filtering, which we keep.
        md = sample.metadata or {}
        # The challenge ``Evals/`` ``index`` is the dataset row index for
        # the v5 subset. Upstream LCB exposes it on Sample.metadata when
        # ``save_metadata`` is enabled; we set it ourselves below.
        idx = md.get("__calibration_index")
        if idx is not None and int(idx) not in self._kept_indices:
            return False
        return super().sample_filter(sample)

    def record_to_sample(self, record: Dict[str, Any]):  # type: ignore[override]
        """Same as upstream, but stamp the dataset row position onto
        ``metadata.__calibration_index`` so ``sample_filter`` can match
        on it. Upstream LCB doesn't currently expose a per-row index in
        the Sample, so we add one here."""
        sample = super().record_to_sample(record)
        if sample.metadata is None:
            sample.metadata = {}
        # Use the upstream dataset's ``evaluation_sample.problem_id`` if
        # present (stable across runs); fall back to a monotonic counter.
        eval_sample = sample.metadata.get("evaluation_sample") or {}
        problem_id = eval_sample.get("question_id") or eval_sample.get("problem_id")
        if problem_id is not None:
            sample.metadata["__calibration_index"] = problem_id
        else:
            # Fall back: the calibration index is the per-row offset assigned
            # by the loader. We let the upstream adapter handle ordering and
            # rely on a tracker incremented in ``load_dataset``.
            sample.metadata.setdefault("__calibration_index", self._row_counter_next())
        return sample

    # --- Row-counter fallback for datasets without a stable problem_id ----
    _row_counter: int = -1  # class-level, reset per adapter instance via __init__

    def _row_counter_next(self) -> int:
        self._row_counter = (self._row_counter or -1) + 1
        return self._row_counter

    def describe_pruning(self) -> dict[str, Any]:
        """Diagnostic — surfaced by compare_runs for the report header."""
        return {
            "strategy": self._strategy_name,
            "prune_ratio": self._prune_ratio,
            "rng_seed": self._rng_seed,
            "calibration_benchmark": self._calibration_benchmark,
            "n_kept": len(self._kept_indices),
        }
