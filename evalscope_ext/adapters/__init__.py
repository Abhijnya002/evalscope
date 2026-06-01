"""Pruned dataset adapters — register with evalscope via @register_benchmark.

Importing this package side-imports each adapter module so the
``@register_benchmark`` decorators run at framework startup. The
adapters are then discoverable as ``--datasets live_code_bench_pruned``,
``--datasets aa_lcr_pruned``, and ``--datasets mmmu_encoder_probe`` via
evalscope's CLI.
"""

from . import aa_lcr_pruned, live_code_bench_pruned, mmmu_encoder_probe  # noqa: F401

__all__ = ["aa_lcr_pruned", "live_code_bench_pruned", "mmmu_encoder_probe"]
