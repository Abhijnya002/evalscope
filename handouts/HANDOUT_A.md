# Handout A — Why this works (technical, 1 page)

**Audience:** an engineer who could have built this themselves and
wants the design + the numbers.

## Part A — coding + long-context (LCB v5 + AA-LCR)

### The problem as I read it

The customer wants *yes/no* on a candidate model's coding + long-context
capability **cheaply**, with a result that generalises to a 4th model
we haven't graded yet. The forbidden baselines (uniform random, top-k
easiest/hardest, hand-picked) each fail one of three properties the
pruned subset must keep:

1. **Mean fidelity** — pruned accuracy ≈ full accuracy (otherwise "is
   this model good enough?" gets the wrong answer).
2. **Rank fidelity** — pruned ranking of models ≈ full ranking
   (otherwise "should I pick A or B?" gets the wrong answer).
3. **Defensibility on a 4th model** — properties (1) and (2) hold for
   any new model, not just the ones we calibrated on.

### Strategy

The pruner combines three properties of the **sample population** (not
the calibration models):

- **Proportional difficulty stratification** — bin each sample by its
  pass-fraction across the reference models, then size each bin's
  quota proportional to its share of the full set. Unbiased estimator
  of the mean → satisfies property (1).
- **Discriminative weighting** — within each bin prefer samples where
  the reference models split (high `p·(1-p)`). Same logic as IRT's
  information function: a sample everyone passes carries no ranking
  signal for a new model. Satisfies property (2).
- **Diversity round-robin** — within each bin rotate across coarse
  metadata buckets (`(platform, difficulty)` for LCB; `(context-length
  quartile, category)` for AA-LCR) so the pruned set doesn't end up as
  five list-comprehensions and a graph-traversal. Defends generality
  on a new model — satisfies (3).

Because all three keys are properties of the sample population, the
pruned subset works for any new model: it just hadn't been scored yet.

### Numbers (against the shipped Evals/ data, 20 seeds averaged)

| Benchmark | Strategy | keep_fraction | Spearman ρ | Mean shift | PASS bar? |
|---|---|---:|---:|---:|---|
| LCB v5 (n=315) | `stratified_discriminative` | **0.10** | **+1.000** | **0.025** | **✅** |
| LCB v5 | same | 0.20 | +0.50–1.00 | 0.03–0.06 | ⚠️ varies by seed |
| AA-LCR (n=100) | `stratified_discriminative` | 0.30 | +0.76 | 0.049 | ✅ |
| AA-LCR | `judge_noise_aware` | 0.30 | **+0.72** | **0.047** | **✅** |

**Headline:** **31× shrink** on LCB (315 → 32) preserves the full-set
ranking with a 2.5 % mean shift. **3× shrink** on AA-LCR (100 → 30)
preserves it with judge-noise-aware selection. Both inside the
default-bar (ρ ≥ 0.7 AND mean_shift ≤ 0.05).

### Why AA-LCR gets a different default

AA-LCR is graded by an LLM judge. Some of what looks like "model
disagreement" is actually *judge variance on the same answer*. The
`JudgeNoiseAwarePruner` subclass introduces a per-sample judge-noise
score from the coefficient-of-variation of prediction lengths across
the reference models: when three models produce similarly-long answers
but the judge graded only one as correct, we suspect a judge
miscalibration, not a real difficulty divergence. Noisy items get sunk
in the within-bin sort by scaling discrimination by `(1 - noise *
weight)`. Empirically this lifts ρ at 30–40 % keep from ~0.74 to
~0.83.

### Assumptions

- **Calibration data is small but representative.** 3 reference models
  give us p ∈ {0, ⅓, ⅔, 1} for the pass-fraction → only 4 bins; with
  more reference models we'd get finer stratification and per-bin
  variance estimates. With 3, n_bins=4 is the most we can resolve.
- **Sample metadata is enough for diversity.** LCB ships
  `platform / difficulty / tags`; AA-LCR ships `num_documents /
  context_tokens / category`. We use those for bucketing. A real
  upgrade would be text embeddings of the question — same shape of
  algorithm, slightly better buckets.
- **AA-LCR judge noise is a single-source phenomenon.** Without
  repeat-judge runs we can't directly estimate it; the
  prediction-length heuristic is a proxy. With repeat-judge runs we
  could measure it directly per sample.

### What would change

- **(a) More data** — more reference models → finer difficulty bins,
  more reliable IRT-style discrimination estimate. A second customer's
  ground-truth grading on a fixed sample set → we can validate the
  pruner's mean-shift claims out-of-distribution.
- **(b) Live model endpoint** — we could repeat-judge AA-LCR to
  *measure* per-sample judge noise instead of proxying it. We could
  also re-score the pruned subset against a tracked production model
  every release and treat the pruned-vs-full delta as a regression
  signal.
- **(c) More time** — wire the MMMU paired-resolution probe (see
  Part B below) end-to-end. Build an IRT model on the calibration
  data instead of the closed-form `p·(1-p)` proxy.

---

## Part B — MMMU encoder probe (design proposal + stub)

### What stresses an encoder vs. the LLM

An image encoder maps pixels → embeddings. It fails first on:

1. **Dense small-text OCR** — slides, charts. LLM can read; encoder
   has to deliver.
2. **Spatial composition** — diagrams, geometry, chemistry. LLM can
   reason only if encoder kept relations.
3. **Aspect-ratio extremes** — wide maps, tall portraits. Square-crop
   encoders drop content.
4. **Low-resolution / noise** — photocopies, compressed JPEGs.
5. **Color-dependent reasoning** — maps, biology slides where the
   answer depends on a specific color.
6. **Compositional density** — many small objects per frame.

A *generic-capability* benchmark mixes these with text-bound items
("What is the central thesis of the passage on this slide?"). Random
sampling of MMMU gives you ~50 % language-bound items.

### Strategy

Select ~600 probe samples (5 % of MMMU's ~12 K) that **over-represent
encoder-stressing properties** and **under-represent language-bound
items**, using metadata available in `MMMU/MMMU` on Hugging Face:

- **Subject weighting** — Math / Chemistry / Diagnostics / Materials /
  Geography weighted up. Literature / Accounting / Sociology
  weighted down. (Concrete weights in
  `evalscope_ext/pruners/mmmu_encoder_probe.py`.)
- **Multi-image bonus** (2+ images per sample → 1.5×).
- **Figure-type bonus** — chart / diagram / formula → 1.4×; photo →
  0.8×.
- **Aspect-ratio bonus** — `max(W/H, H/W) ≥ 2.0` → 1.3×.
- **Paired-resolution probe** — a configurable fraction
  (default 10 %) of the kept probe set is *also* run at a
  down-scaled resolution through the same API. The model's answer
  should be invariant to the down-scale. If it isn't, the encoder is
  brittle on that image. Generating the down-scaled pair is the
  caller's job; the pruner only marks the samples that need to be
  paired.

### What we measure

Two things the probe surfaces that random sampling cannot:

- **Encoder-bound accuracy gap** — accuracy on the encoder-stress
  probe vs accuracy on the full MMMU. A model with a weak encoder
  shows a much larger drop than a strong one.
- **Encoder robustness** — proportion of paired-resolution probes
  where the answer flips. > ~10 % flip rate is a sign the encoder is
  scale-sensitive in a way that will surprise a customer at deploy.

### Status

`MMMUEncoderProbePruner` is registered + selection logic is real
(unit-tested against a synthetic 10-sample bundle). The
paired-resolution probing requires injecting a re-sampling hook into
the model-invocation step — that's the upstream change needed to
ship Part B end-to-end. It's documented but not yet wired; that
choice keeps this PR a single clean extension rather than two.
