"""Pruned dataset adapters — register with evalscope via @register_benchmark.

Importing this package side-imports each adapter module so the
``@register_benchmark`` decorators run at framework startup. The
adapters are then discoverable as ``--datasets live_code_bench_pruned``
and ``--datasets aa_lcr_pruned`` via evalscope's CLI.
"""

from . import aa_lcr_pruned, live_code_bench_pruned  # noqa: F401 — side-import for registration

__all__ = ["aa_lcr_pruned", "live_code_bench_pruned"]
