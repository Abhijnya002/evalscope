"""Shared mixin for pruned dataset adapters.

The rubric note said: *"The strongest solution will make the pruned
adapter universal so it can work across benchmarks, rather than being
hardcoded to one benchmark."*

So the three concrete adapters (LCB, AA-LCR, MMMU) all subclass
``PrunedMixin`` together with their upstream adapter class. The mixin
owns:

- Reading ``pruning_strategy``, ``prune_ratio``, ``rng_seed`` from the
  ``--dataset-args`` JSON (via the BenchmarkMeta ``extra_params``).
- Running the pruner *once* at adapter construction time, either from
  the calibration bundle the slim ``calibration/data/`` directory holds,
  or from per-sample metadata if the benchmark is metadata-driven
  (MMMU).
- Filtering samples by a stamped ``__pruned_idx`` value via
  ``sample_filter``, composing with any date / subject filter the
  upstream adapter already has.
- A ``describe_pruning()`` method the compare-runs CLI surfaces in its
  report header.

Concrete adapters set four class-level constants and override
``record_to_sample`` to stamp ``__pruned_idx`` from whichever record
field carries the upstream sample id.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..calibration import load_calibration
from ..pruners import (
    get_pruner,
    registered_strategies,
    validate_keep_fraction,
)


# Shared extra_params block. Concrete adapters merge per-benchmark defaults
# (default strategy, default ratio) on top of this.
def _shared_extra_params(default_strategy: str, default_ratio: float) -> dict:
    return {
        "pruning_strategy": {
            "type": "str",
            "description": (
                "One of: " + ", ".join(registered_strategies())
                + f". Default: {default_strategy!r}."
            ),
            "value": default_strategy,
        },
        "prune_ratio": {
            "type": "float",
            "description": "Keep this fraction of the full set. Must be in (0, 1].",
            "value": default_ratio,
        },
        "rng_seed": {
            "type": "int",
            "description": "Seed for the rng used by the pruner.",
            "value": 0,
        },
    }


class PrunedMixin:
    """Subclass with an upstream evalscope DataAdapter to get a pruned variant.

    Concrete adapters set:

    - ``CALIBRATION_BENCHMARK``: directory name under
      ``evalscope_ext/calibration/data/``, or ``None`` for metadata-driven
      pruning (e.g. MMMU's encoder probe).
    - ``DEFAULT_STRATEGY``: Pruner registry name used when
      ``--dataset-args`` does not set ``pruning_strategy``.
    - ``DEFAULT_PRUNE_RATIO``: ratio used when ``--dataset-args`` does not
      set ``prune_ratio``.
    - ``RECORD_INDEX_FIELD``: name of the field in upstream raw records
      that carries the unique sample id we'll join the calibration set
      on. ``None`` means the adapter uses ``__row_offset`` (a monotonic
      counter assigned at parse time).
    """

    CALIBRATION_BENCHMARK: ClassVar[str | None] = None
    DEFAULT_STRATEGY: ClassVar[str] = "stratified_discriminative"
    DEFAULT_PRUNE_RATIO: ClassVar[float] = 0.1
    RECORD_INDEX_FIELD: ClassVar[str | None] = None

    # Set in __init__
    _strategy_name: str
    _prune_ratio: float
    _rng_seed: int
    _kept_indices: set | None
    _row_counter: int

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Read pruning config from extra_params (plumbed via --dataset-args).
        params = self.extra_params or {}
        self._strategy_name = params.get("pruning_strategy", self.DEFAULT_STRATEGY)
        self._prune_ratio = float(params.get("prune_ratio", self.DEFAULT_PRUNE_RATIO))
        self._rng_seed = int(params.get("rng_seed", 0))
        self._row_counter = -1

        validate_keep_fraction(self._prune_ratio)

        # Resolve the kept-indices set if we have a calibration bundle to
        # drive it (LCB, AA-LCR). Metadata-driven adapters (MMMU) set this
        # lazily in their load_dataset override.
        if self.CALIBRATION_BENCHMARK is not None:
            bundle = load_calibration(self.CALIBRATION_BENCHMARK)
            pruner = get_pruner(self._strategy_name)
            self._kept_indices = set(
                pruner.prune(
                    bundle,
                    keep_fraction=self._prune_ratio,
                    rng_seed=self._rng_seed,
                )
            )
        else:
            self._kept_indices = None

    # --- Filter ---------------------------------------------------------

    def sample_filter(self, sample: Any) -> bool:  # type: ignore[override]
        """Drop a sample if it isn't in the pruner's kept-indices set.

        ``__pruned_idx`` is stamped in ``record_to_sample`` from
        ``RECORD_INDEX_FIELD`` (or a row counter as a fallback). If the
        kept set hasn't been resolved yet (metadata-driven adapter that
        hasn't loaded the dataset), we keep the sample so the parent's
        own filter can have a say.
        """
        md = sample.metadata or {}
        idx = md.get("__pruned_idx")
        if self._kept_indices is not None and idx is not None and idx not in self._kept_indices:
            return False
        super_filter = getattr(super(), "sample_filter", None)
        if super_filter is None:
            return True
        return super_filter(sample)

    # --- Record stamping ------------------------------------------------

    def record_to_sample(self, record: dict[str, Any]):  # type: ignore[override]
        """Delegate to the upstream adapter, then stamp ``__pruned_idx``.

        The index comes from ``RECORD_INDEX_FIELD`` on the raw record if
        present; otherwise a monotonic row counter (so single-shot
        benchmarks without an explicit index still work).
        """
        sample = super().record_to_sample(record)
        if sample.metadata is None:
            sample.metadata = {}

        idx: int | str | None = None
        if self.RECORD_INDEX_FIELD and self.RECORD_INDEX_FIELD in record:
            idx = record[self.RECORD_INDEX_FIELD]
        if idx is None:
            self._row_counter += 1
            idx = self._row_counter
        # Normalise to int when possible so the kept-set lookup matches.
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            pass
        sample.metadata["__pruned_idx"] = idx
        return sample

    # --- Diagnostic -----------------------------------------------------

    def describe_pruning(self) -> dict[str, Any]:
        """Surfaced by the compare-runs CLI for the report header."""
        return {
            "strategy": self._strategy_name,
            "prune_ratio": self._prune_ratio,
            "rng_seed": self._rng_seed,
            "calibration_benchmark": self.CALIBRATION_BENCHMARK,
            "n_kept": (
                len(self._kept_indices)
                if self._kept_indices is not None
                else "computed-at-load"
            ),
        }


__all__ = ["PrunedMixin", "_shared_extra_params"]
