# About the handouts

This folder is a plain walkthrough of what the two handouts say. The
handouts themselves live one folder up, under `handouts/`. Pick the
one that fits the reader.

## Handout A is for engineers

Read this if you write code or check the math. It covers five things.

**Why this pruner is here at all.** The simple options each fail in
their own way. Random sampling gives noisy numbers. Pick the easiest
30 and every model looks good. Pick the hardest 30 and every model
looks bad. None of those keep the order between models, which is the
answer a sales call actually wants.

**The three signals that drive the pick.** First, difficulty: how
often the three test models pass a sample. Second, how much the
models disagree on the sample. A sample where all three pass tells
you nothing about a fourth model. A sample where two pass and one
fails tells you a lot. Third, light metadata like problem topic or
context length, used to keep the pick varied. Each signal is a
property of the test set, not the test models. That matters. The
goal is a small set that still works for a model we have not graded
yet.

**The numbers we hit on the real data.** Live-Code-Bench shrinks from
315 samples to 32. The model order on the small set still matches
the model order on the full set. AA-LCR shrinks from 100 to 30. AA-LCR
is harder to prune because the judge is an LLM and the judge adds
noise. We wrote a second pruner that tries to spot judge noise from
the length of the model's answer, and sinks those samples lower in
the pick.

**The MMMU plan.** This is forward looking. We pick samples that
stress the image encoder, not the language brain. Think dense
diagrams, wide maps, charts with small numbers, questions with two
or three images. There is also a small set of paired probes at two
image sizes. The model should give the same answer at both sizes. If
it does not, the encoder is shaky.

**What we would do with more time.** Repeat-judge the AA-LCR samples
so we can measure judge noise directly instead of guessing it. Run
the pruner against a fourth, unseen model to confirm the subset
travels. Wire the paired-resolution probe all the way through the
model call.

If you want to argue with the design, Handout A is the place to
start.

## Handout B is for the rest of the team

Read this if you sit in sales, product, or deployment. It covers
four things.

**What this changes for a customer call.** The full eval on a
candidate model took hours of compute. The pruned eval takes a small
fraction of that. The report at the end says PASS or FAIL on whether
the smaller run still tells the truth. That is day one of an
engagement instead of day three.

**The three lines you would type tomorrow.** One for the full
reference run. One for the pruned run, with the strategy and the
keep ratio. One for the compare report. The report puts each model's
full and pruned accuracy next to each other, with a rank-correlation
number and a max-shift number. PASS means rank correlation at or
above 0.7 and max shift at or below five percent.

**Why the multimodal probe beats random sampling.** A random five
percent of MMMU is mostly text-on-slides. You can read the question
from the image and answer with words. That tests the language brain,
not the image encoder. The probe targets the samples where the
encoder has to do real work.

**Why a customer-facing PM should care.** Two reasons. First, faster
answers move sales cycles. Second, honest answers protect you from
quoting a number the full eval would have argued with.

Handout B is short on purpose. It is the page you would send to a
sales lead before a call.

## Where to read next

If you want to follow the code after reading the handouts, this is
the order that worked for us:

1. `evalscope_ext/calibration/loader.py` reads the test-model scores
   into one normalised view. Start here so the rest reads cleanly.
2. `evalscope_ext/pruners/stratified.py` is the main pruner. The
   docstring at the top of the file is the design in thirty lines.
3. `evalscope_ext/pruners/judge_noise_aware.py` is the AA-LCR
   variant, with the prediction-length noise score.
4. `evalscope_ext/adapters/*_pruned.py` are the three wrappers that
   plug into the evalscope CLI.
5. `evalscope_ext/tools/compare_runs.py` is the report tool from the
   run contract.

The tests under `evalscope_ext/tests/` lock down the claims in
Handout A. If a number in a handout changes, run the suite to check
the new number is real.
