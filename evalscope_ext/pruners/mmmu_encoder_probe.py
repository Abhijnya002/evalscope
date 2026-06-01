"""Part B — MMMU encoder probe (design + executable stub).

The challenge's Part B asks: if a Cerebras prospect extends into
multimodal next quarter, how would we *cheaply* tell them whether a
candidate model's image **encoder** is good enough — distinct from
generic capability?

This is a *forward-looking* exercise. We don't have multi-model MMMU
review data to calibrate on (only one model — `glm-4.5v-fp8`); we have
to reason about the ~12 K-sample full MMMU dataset on HuggingFace.

## What stresses an encoder vs. the LLM

An image encoder maps pixels to embeddings. It fails first on:

1. **Small text / dense OCR.** Slides, charts, dense academic figures.
   The LLM can read text fine; the encoder has to *deliver* the text
   tokens for the LLM to read. A weak encoder loses the small print.
2. **Spatial / compositional reasoning.** Diagrams with arrows + labels,
   chemistry structures, geometry problems. The LLM can reason about a
   diagram only if the encoder preserved the relations between
   sub-components.
3. **Aspect-ratio extremes.** Many encoders downsample to a square
   ~224 / 384 / 448 grid. Wide panoramas or tall portraits get squished
   or letterboxed — the encoder loses the parts that fall outside the
   crop.
4. **Low-resolution / noisy input.** Photocopied diagrams, scans,
   compressed JPEGs. The encoder's denoising is what determines if the
   LLM gets a clean signal.
5. **Color-dependent reasoning.** Maps, schematics, biology slides where
   the answer depends on a specific color. A degraded encoder collapses
   the colour channels.
6. **Compositional density.** Many small objects in one frame
   (microscope images, satellite shots). The encoder has finite spatial
   resolution; a dense scene blows past it.

A *generic-capability* benchmark mixes these with samples that look
hard but are actually language-bound ("What is the central thesis of
this paragraph quoted on the slide?" — encoder reads ~40 words once,
LLM does the work).

## Probe strategy

The pruner here picks ~500 probe samples from the full ~12 K MMMU
dataset that *over-represent encoder-stressing properties* and
*under-represent language-only items*. Concretely (heuristics applied
to the HuggingFace `MMMU/MMMU` dataset records — see the docstring
for `select_probe`):

- **subject filter** — drop subjects whose accuracy on the shipped
  glm-4.5v-fp8 run is essentially purely text (Literature, Music
  to a lesser extent). Over-sample Math, Chemistry, Diagnostics and
  Materials — these are the densest-image subjects.
- **multi-image bonus** — MMMU samples can carry up to four images;
  multi-image samples stress the encoder's per-token budget. Weight
  them ~2x.
- **figure-type bonus** — many samples have a "figure type" tag
  (chart / diagram / photo / formula / map). Up-weight chart +
  diagram + formula (encoder-bound); down-weight photo (mostly
  language-bound).
- **aspect-ratio bonus** — using the dataset's image dimensions,
  oversample images with extreme aspect ratios (>2:1 or <1:2). Those
  are the ones encoders most often resize-crop badly.
- **size-stratified contrast pairs** — for a small subset of probe
  samples, *also* request the same sample at a downscaled resolution
  through the same API to measure encoder *robustness* (the model's
  answer should be the same; if it changes, the encoder is brittle
  at small input sizes). This needs paired probing — see
  ``ProbePair`` below.

Because we don't have per-sample image features in the shipped slim
calibration data, the implementation here is a *stub* — it shows the
selection logic and the data shape that the full version would
consume. Handout A spells out exactly what extra inputs make this
production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..calibration import CalibrationBundle
from .base import Pruner, register_pruner, validate_keep_fraction

# Encoder-stress-relevant signal weights. Documented + tunable.
_SUBJECT_WEIGHTS = {
    # Math / chemistry / diagnostics / materials — image-bound subjects.
    "Math": 1.5,
    "Chemistry": 1.5,
    "Diagnostics_and_Laboratory_Medicine": 1.5,
    "Materials": 1.4,
    "Geography": 1.3,
    "Clinical_Medicine": 1.3,
    "Architecture_and_Engineering": 1.3,
    "Mechanical_Engineering": 1.3,
    "Physics": 1.3,
    "Electronics": 1.3,
    "Biology": 1.2,
    "Computer_Science": 1.1,
    "Pharmacy": 1.1,
    "Public_Health": 1.0,
    "Energy_and_Power": 1.0,
    # Mid — diagrams + photos.
    "Agriculture": 1.0,
    "Design": 1.0,
    "Art": 0.9,
    "Art_Theory": 0.9,
    # Language-heavy subjects — under-sample.
    "Accounting": 0.6,
    "Economics": 0.6,
    "Finance": 0.6,
    "Manage": 0.6,
    "Marketing": 0.6,
    "Sociology": 0.6,
    "Psychology": 0.6,
    "Literature": 0.4,
    "Music": 0.6,
    "History": 0.6,
}


@dataclass(frozen=True)
class ProbePair:
    """A probe at two resolutions of the same image, asked of the same model.

    The model's answer should be invariant to the down-scale. If it
    isn't, the encoder is brittle on this image. Generating the
    down-scaled pair is the caller's job — the pruner only marks the
    samples that should be paired.
    """

    sample_index: int
    full_resolution: tuple[int, int]
    downscale_resolution: tuple[int, int]


def _sample_weight(metadata: dict[str, Any]) -> float:
    """Encoder-stress weight for one sample's metadata.

    Combines subject, image count, image type, and aspect ratio. All
    inputs are graceful — missing metadata defaults to 1.0 (neutral)
    so the pruner still functions on a partial calibration dump.
    """
    w = 1.0
    subj = metadata.get("subject") or metadata.get("subfield")
    if subj in _SUBJECT_WEIGHTS:
        w *= _SUBJECT_WEIGHTS[subj]

    # Multi-image samples stress encoder per-token budget.
    n_imgs = metadata.get("num_images") or len(metadata.get("images") or []) or 1
    if n_imgs >= 2:
        w *= 1.5

    # Figure type (chart / diagram / formula / photo / map).
    ft = metadata.get("image_type") or metadata.get("figure_type") or ""
    if isinstance(ft, str):
        ft_low = ft.lower()
        if any(k in ft_low for k in ("chart", "diagram", "formula", "schema", "graph")):
            w *= 1.4
        elif "photo" in ft_low or "photograph" in ft_low:
            w *= 0.8

    # Aspect ratio.
    width = metadata.get("width") or metadata.get("image_width")
    height = metadata.get("height") or metadata.get("image_height")
    if width and height:
        ar = max(width / height, height / width)
        if ar >= 2.0:
            w *= 1.3

    return w


@register_pruner
class MMMUEncoderProbePruner(Pruner):
    """Selects MMMU samples that stress image encoders specifically.

    Configuration:

    - ``probe_pair_fraction``: fraction of the probe set that is also
      assigned a paired downscaled-resolution probe. Default 0.1 = a
      tenth of the kept samples get a contrast pair, which is what's
      needed to estimate encoder robustness without doubling cost.

    Status: **stub implementation** for the design proposal in
    Handout A. The selection logic + signal weights are real; the
    integration with the full ~12 K HF dataset assumes the metadata
    fields documented in ``_sample_weight``. The shipped slim
    calibration data only has 22 subject summaries (not per-sample),
    so the unit tests verify the *shape* of the selection rather than
    its concrete output.
    """

    name = "mmmu_encoder_probe"

    def __init__(self, *, probe_pair_fraction: float = 0.1) -> None:
        if not (0.0 <= probe_pair_fraction <= 1.0):
            raise ValueError("probe_pair_fraction must be in [0, 1]")
        self.probe_pair_fraction = probe_pair_fraction
        self.probe_pairs: list[ProbePair] = []  # populated by prune()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "probe_pair_fraction": self.probe_pair_fraction,
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

        # Score each sample, pick the top N by encoder-stress weight.
        weighted = [
            (s.index, _sample_weight(s.metadata or {}))
            for s in bundle.samples
        ]
        target = max(1, round(bundle.n_samples * keep_fraction))
        weighted.sort(key=lambda t: (-t[1], t[0]))
        kept = sorted(idx for idx, _w in weighted[:target])

        # Probe-pair marking — sample evenly from kept indices.
        n_pairs = int(round(len(kept) * self.probe_pair_fraction))
        if n_pairs > 0:
            import random as _random

            rng = _random.Random(rng_seed)
            paired = rng.sample(kept, n_pairs)
            self.probe_pairs = [
                ProbePair(sample_index=i, full_resolution=(0, 0), downscale_resolution=(0, 0))
                for i in paired
            ]
        else:
            self.probe_pairs = []

        return kept
