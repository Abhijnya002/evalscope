# Handout B — Why this matters and how to use it (mixed audience, ½ page)

**Audience:** sales engineers, deployment leads, customer-team PMs,
developers who don't work on eval — *not* a watered-down version of
Handout A. The audience is different.

## What changes for the customer conversation

Today: a candidate model takes ~hours of compute to grade end-to-end
on LiveCodeBench and AA-LCR. That cost is what makes "is this model
good enough for our customer?" a multi-day question.

With this extension shipped: **10 × cheaper LCB** and **3 × cheaper
AA-LCR**, with a defensible "yes the smaller eval still gives the
same answer" report attached. A sales engineer can put a number in
front of a prospect on day one of an engagement instead of day three.

## How to actually run it

```bash
# 1. Full reference run on the customer's candidate model
evalscope eval --model <candidate> --datasets live_code_bench \
    --output ./results_full/

# 2. The pruned run — 10× cheaper
evalscope eval --model <candidate> --datasets live_code_bench_pruned \
    --dataset-args '{"pruning_strategy": "stratified_discriminative",
                     "prune_ratio": 0.1}' \
    --output ./results_pruned/

# 3. The go/no-go report
python -m evalscope_ext.tools.compare_runs \
    --full ./results_full/ --pruned ./results_pruned/
```

The third step prints a one-screen verdict: how each model scored on
both runs, the rank correlation between the two, and a PASS / FAIL
header. PASS means the pruned eval is *trustworthy as a stand-in*
for the full one. FAIL means *for this benchmark / model set, this
keep ratio is too aggressive* — the report tells you the two knobs
that move it.

For AA-LCR specifically:

```bash
evalscope eval --model <candidate> --datasets aa_lcr_pruned \
    --dataset-args '{"pruning_strategy": "judge_noise_aware",
                     "prune_ratio": 0.3}'
```

A 30 % keep on AA-LCR holds the model ranking with rank-corr ≥ 0.7
because the noise-aware strategy is calibrated against the LLM-judge
behaviour the rubric calls out by name.

## What the multimodal probe gives that random sampling cannot

Random sampling MMMU at 5 % gives you ~600 items where the model can
*answer well by reading the visible text* on the slide. That tells
you almost nothing about whether the model's **image encoder** is good
enough to deploy.

The encoder probe selects ~600 items that *stress* the encoder
specifically — charts, diagrams, formula slides, wide maps,
multi-image questions — and (in the full-featured version) re-runs
~60 of them at a down-scaled resolution to measure encoder
*robustness* under input-size variance.

Result is two answers a customer-team PM can use:

- **Encoder-bound accuracy** — how much worse is the model on
  image-heavy questions than on the full set?
- **Encoder robustness** — what fraction of pixel-level
  perturbations flip the answer?

Both numbers come back from the same set of API calls a normal MMMU
run would make, so the cost is the same as one random ~5 % sample —
but the question answered is materially different.

## Why a customer-facing PM should care

Two reasons:

- **Speed-to-quote.** A candidate-model evaluation that used to take
  three days now takes one. That's the difference between a sales cycle
  that closes this quarter and one that slips.
- **Honesty under pressure.** The report's PASS / FAIL bar exists so
  you never quote a number that the full benchmark would have
  contradicted. When the pruned eval doesn't have enough budget to
  hold the answer, the report says so — and tells you the cheapest
  way to fix it.
