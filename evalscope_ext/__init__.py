"""evalscope_ext — benchmark pruning extension for modelscope/evalscope.

This package implements the Cerebras AI Engineer Model Quality &
Performance challenge (Task 2): pruning LiveCodeBench and AA-LCR to the
smallest sample set that still gives a useful good-or-not signal for a
new model, plus a forward-looking design for an MMMU image-encoder
probe.

Submodules:

- ``evalscope_ext.calibration`` — load the shipped per-sample reviews
  for the three reference models into a normalized form the pruners
  consume.
- ``evalscope_ext.pruners`` — pruning strategies (difficulty + IRT-style
  discrimination + diversity; AA-LCR judge-noise robustification).
- ``evalscope_ext.adapters`` — pruned dataset adapters registered with
  evalscope via the ``@register_benchmark`` decorator. These wrap the
  upstream ``live_code_bench`` / ``aa_lcr`` / ``mmmu`` adapters.
- ``evalscope_ext.tools`` — the ``compare_runs`` CLI for the rubric's
  run contract.

Pinned to modelscope/evalscope commit ``e9d42d8`` — see
``evalscope_ext/README.md`` for the rationale and the upgrade path.
"""

__version__ = "0.1.0"

UPSTREAM_PINNED_SHA = "e9d42d8b6a8dcb937e042ba905e36eb05171ae0d"
