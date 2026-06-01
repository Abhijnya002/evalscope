# evalscope_ext — benchmark pruning extension

Cerebras AI Engineer Model Quality & Performance challenge — **Task 2**.

This package extends [modelscope/evalscope](https://github.com/modelscope/evalscope)
with pruned-benchmark variants that let a sales engineer or a deployment
lead score a candidate model on a **small, defensible subset** of
LiveCodeBench / AA-LCR and still get a reliable "good enough?" signal.

> **Pinned upstream commit:** `e9d42d8b6a8dcb937e042ba905e36eb05171ae0d`
> (latest at fork time — 2026-05-29).
>
> The evalscope adapter API is still evolving; if you bump upstream,
> rerun `pytest evalscope_ext/tests/` to verify the
> `@register_benchmark` decorator signature and `DefaultDataAdapter`
> base class haven't shifted under us.

## What it adds

| Dataset name | Wraps | What it does |
|---|---|---|
| `live_code_bench_pruned` | `live_code_bench` | Selects ~10–30 % of LCB v5 using the stratified-discriminative pruner. |
| `aa_lcr_pruned` | `aa_lcr` | Selects ~30–50 % of AA-LCR using a judge-noise-aware variant. |
| `mmmu_encoder_probe` | `mmmu` | **Part B — design + stub.** Selects an encoder-stressing probe set out of the full HF MMMU. |

## Run contract

The rubric specifies this exact command shape. It works as-is once this
package is on the import path:

```bash
# Full reference run (one of the three shipped models)
evalscope eval --model gpt-oss-120b --datasets live_code_bench --output ./results_full/

# Pruned run with our strategy + ratio
evalscope eval --model gpt-oss-120b --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "stratified_discriminative", "prune_ratio": 0.1}' \
    --output ./results_pruned/

# Divergence report — Spearman rank-corr, top-k retention, mean shift
python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
```

## Why not just sample uniformly at random?

The rubric explicitly forbids uniform random, top-k easy/hard, and
hand-picked. Each is dishonest in a specific way:

- **Random** has variance proportional to `1/√n`. At a 10 % prune ratio,
  the standard error on the mean is ~3× larger than the full-set's;
  worse, it picks no samples from rare difficulty bins, so a model that
  is good on average but poor on a specific class of inputs passes
  silently.
- **Top-k easy/hard** is biased. Pruning to the 30 easiest items makes
  every model look good; pruning to the 30 hardest makes every model
  look bad. Neither preserves the *ranking* between models.
- **Hand-picked** is not reproducible across customers / over time.

The strategy in this repo combines three properties of the *samples*
(not the three calibration models):

1. **Difficulty stratification** — bin by fraction of reference models
   that pass; sample uniformly across bins.
2. **Discriminative power** — within each bin, prefer samples where the
   reference models disagree (i.e. they actually separate models).
3. **Diversity** — within each bin, prefer samples that span the
   metadata space (problem topic, prompt length, etc.) so we don't pick
   five list-comprehension problems.

Because items 1–3 are properties of the sample population, the pruned
set generalises to a fourth, unseen model. This is the defensibility
property the rubric calls out.

## Package layout

```
evalscope_ext/
├── README.md
├── __init__.py
├── calibration/      # load shipped predictions + reviews from Evals/
├── pruners/
│   ├── base.py        # Pruner ABC + selection-quality metrics
│   ├── stratified.py  # difficulty + discrimination + diversity
│   └── judge_noise_aware.py  # AA-LCR variant
├── adapters/         # @register_benchmark adapters for each pruned variant
├── tools/
│   └── compare_runs.py
└── tests/
```

## Installing

The extension lives inside the evalscope fork; installing evalscope
from this repo's root pulls it in as part of the `evalscope` package
ecosystem (`pip install -e .` from the repo root). The adapters
auto-register because `evalscope/benchmarks/*/__init__.py` glob-imports
their modules at framework startup.

## Handouts

The technical and mixed-audience handouts live at
[`handouts/HANDOUT_A.md`](../handouts/HANDOUT_A.md) and
[`handouts/HANDOUT_B.md`](../handouts/HANDOUT_B.md).
