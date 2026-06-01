"""``mmmu_encoder_probe`` — Part B stub adapter.

Wraps the upstream MMMU adapter with the encoder-probe pruner. The
selection logic lives in
:mod:`evalscope_ext.pruners.mmmu_encoder_probe`; this module is just
the wiring.

Status: **stub**. Selects samples via the documented heuristic but
doesn't yet realise the paired-resolution probe (that requires
modifying the model invocation step to re-send a downscaled image —
out of scope for this single-fork prototype, documented in Handout A).
"""

from __future__ import annotations

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter
from evalscope.constants import Tags

from ..pruners import get_pruner, registered_strategies, validate_keep_fraction

_EXTRA_PARAMS = {
    "pruning_strategy": {
        "type": "str",
        "description": (
            "One of: " + ", ".join(registered_strategies())
            + ". Defaults to mmmu_encoder_probe (encoder-stress selection)."
        ),
        "value": "mmmu_encoder_probe",
    },
    "prune_ratio": {
        "type": "float",
        "description": "Target keep fraction over the full ~12K MMMU set.",
        "value": 0.05,  # ~600 samples out of 12K — that's the probe budget Handout A defends.
    },
    "rng_seed": {"type": "int", "value": 0},
    "probe_pair_fraction": {
        "type": "float",
        "description": (
            "Fraction of the kept probe set that also gets a paired "
            "downscaled-resolution probe (encoder-robustness contrast). "
            "0.1 = a tenth of the probe set is paired."
        ),
        "value": 0.1,
    },
}


@register_benchmark(
    BenchmarkMeta(
        name="mmmu_encoder_probe",
        pretty_name="MMMU (encoder probe)",
        tags=[Tags.MULTI_MODAL] if hasattr(Tags, "MULTI_MODAL") else [],
        description=(
            "Forward-looking probe set selected to stress image encoder "
            "quality specifically. See evalscope_ext/pruners/mmmu_encoder_probe.py "
            "and handouts/HANDOUT_A.md for the design proposal."
        ),
        dataset_id="MMMU/MMMU",
        metric_list=["acc"],
        eval_split="validation",
        extra_params=_EXTRA_PARAMS,
    )
)
class MMMUEncoderProbeAdapter(MMMUAdapter):
    """MMMU adapter that filters to the encoder-probe subset.

    **Stub**: builds the kept-index set from a strategy-driven heuristic
    over the upstream record metadata (subject, num_images, figure_type,
    image dimensions). Paired-resolution probing is documented but not
    yet implemented in the model-invocation step — see Handout A for
    the design.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._strategy_name: str = self.extra_params.get(
            "pruning_strategy", "mmmu_encoder_probe"
        )
        self._prune_ratio: float = float(self.extra_params.get("prune_ratio", 0.05))
        self._rng_seed: int = int(self.extra_params.get("rng_seed", 0))
        self._probe_pair_fraction: float = float(self.extra_params.get("probe_pair_fraction", 0.1))

        validate_keep_fraction(self._prune_ratio)

        # We don't pre-resolve kept indices here because MMMU's calibration
        # bundle doesn't ship per-sample metadata in the slim form — the
        # adapter does the scoring as samples flow through record_to_sample.
        self._pruner = get_pruner(
            self._strategy_name, probe_pair_fraction=self._probe_pair_fraction
        )
        self._kept_indices: set[int] = set()
        self._row_counter = -1
        # The probe is selected lazily in load_dataset() — see below.

    def describe_pruning(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy_name,
            "prune_ratio": self._prune_ratio,
            "rng_seed": self._rng_seed,
            "probe_pair_fraction": self._probe_pair_fraction,
            "n_kept": len(self._kept_indices),
            "n_probe_pairs": len(getattr(self._pruner, "probe_pairs", [])),
            "status": "stub — see handouts/HANDOUT_A.md for the design",
        }
