"""``mmmu_encoder_probe`` - Part B encoder-stress probe.

Uses the same universal ``PrunedMixin`` as the other two adapters,
with one twist: MMMU has no per-sample calibration data (we only have
one reference model's outputs, not three). So the kept-indices set is
**computed at load time from each sample's metadata** rather than
read from a pre-built CalibrationBundle.

The override is in ``_resolve_kept_indices()``, which iterates the
upstream MMMU dataset once after it loads, scores each sample by the
encoder-stress weight, and picks the top ``prune_ratio`` fraction
globally. After that, ``sample_filter`` (inherited from the mixin)
does the actual drop.
"""

from __future__ import annotations

from typing import Any

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter

from ..pruners.mmmu_encoder_probe import _sample_weight
from ._base import PrunedMixin, _shared_extra_params

try:
    from evalscope.constants import Tags
    _TAGS = [Tags.MULTI_MODAL] if hasattr(Tags, "MULTI_MODAL") else []
except Exception:  # pragma: no cover
    _TAGS = []

_EXTRA_PARAMS = {
    **_shared_extra_params("mmmu_encoder_probe", 0.05),
    "probe_pair_fraction": {
        "type": "float",
        "description": (
            "Fraction of the kept probe set that also gets a paired "
            "downscaled-resolution probe. 0.1 = a tenth of the probe set "
            "is paired."
        ),
        "value": 0.1,
    },
}


@register_benchmark(
    BenchmarkMeta(
        name="mmmu_encoder_probe",
        pretty_name="MMMU (encoder probe)",
        tags=_TAGS,
        description=(
            "MMMU subset selected to stress the image encoder rather than "
            "the language brain. See evalscope_ext/pruners/mmmu_encoder_probe.py "
            "and handouts/HANDOUT_A.md for the design."
        ),
        dataset_id="MMMU/MMMU",
        metric_list=["acc"],
        eval_split="validation",
        extra_params=_EXTRA_PARAMS,
    )
)
class MMMUEncoderProbeAdapter(PrunedMixin, MMMUAdapter):
    """MMMU adapter that runs the encoder-stress probe at load time.

    Unlike the LCB and AA-LCR pruned adapters, this one cannot resolve
    the kept set at construction time because the kept set is a
    function of the full MMMU dataset's per-sample metadata. So we
    override ``load_dataset`` to do the one-time pass after upstream
    loads.
    """

    DEFAULT_STRATEGY = "mmmu_encoder_probe"
    DEFAULT_PRUNE_RATIO = 0.05  # ~600 of the full 12K MMMU set
    CALIBRATION_BENCHMARK = None  # metadata-driven, not calibration-driven
    RECORD_INDEX_FIELD = "id"  # MMMU records use 'id' as the stable per-sample key

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        params = self.extra_params or {}
        self._probe_pair_fraction = float(params.get("probe_pair_fraction", 0.1))
        self._probe_pairs: list[Any] = []

    def load_dataset(self):  # type: ignore[override]
        """Load full MMMU upstream, then prune in place.

        The probe scoring is deterministic for a given dataset shape, so
        repeat calls on the same upstream snapshot pick the same probe.
        ``rng_seed`` controls only the paired-resolution sampling, which
        is the only stochastic part.
        """
        ds_dict = super().load_dataset()

        # Score every sample. Each sample carries the same metadata fields
        # the pruner's _sample_weight() expects (subject, num_images,
        # figure_type, width/height) - those come from MMMU's HF schema.
        scored = []
        for subset_name, dataset in ds_dict.items():
            for sample in dataset:
                meta = dict(sample.metadata or {})
                # Make sure subject is set; MMMU subset name often IS the subject.
                meta.setdefault("subject", subset_name)
                w = _sample_weight(meta)
                # Use the stamped __pruned_idx as the key; falls back to id() below.
                key = (subset_name, meta.get("id") or meta.get("__pruned_idx") or id(sample))
                scored.append((key, w, sample))

        if not scored:
            return ds_dict

        # Pick the top-N by score globally across subjects (so a hard
        # subject can spend more of the budget than an easy one).
        scored.sort(key=lambda t: -t[1])
        n_keep = max(1, round(len(scored) * self._prune_ratio))
        kept_keys = {key for key, _, _ in scored[:n_keep]}

        # Mark paired-resolution probes (a fraction of the kept set).
        import random as _random
        rng = _random.Random(self._rng_seed)
        n_pairs = int(round(n_keep * self._probe_pair_fraction))
        if n_pairs > 0:
            kept_list = [key for key, _, _ in scored[:n_keep]]
            self._probe_pairs = rng.sample(kept_list, n_pairs)

        # Filter each subset's dataset in place to the kept keys.
        for subset_name, dataset in list(ds_dict.items()):
            keep_samples = [
                s for s in dataset
                if (subset_name, (s.metadata or {}).get("id")
                    or (s.metadata or {}).get("__pruned_idx") or id(s)) in kept_keys
            ]
            # Use the dataset's own factory so type metadata survives.
            try:
                ds_dict[subset_name] = type(dataset)(keep_samples)
            except Exception:
                # Fall back: rebind to a plain list (the evaluator iterates it).
                ds_dict[subset_name] = keep_samples

        # Set kept_indices to a tombstone so sample_filter is a no-op past this
        # point (the filtering already happened above).
        self._kept_indices = set()  # empty -> sample_filter passes everything through

        return ds_dict

    def describe_pruning(self) -> dict[str, Any]:  # type: ignore[override]
        d = super().describe_pruning()
        d["probe_pair_fraction"] = self._probe_pair_fraction
        d["n_probe_pairs"] = len(self._probe_pairs)
        d["status"] = "runnable; paired-resolution scoring pending model endpoint"
        return d
