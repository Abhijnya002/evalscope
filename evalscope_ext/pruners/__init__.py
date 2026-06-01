"""Pruning strategies. Selection is the *policy*; adapters are the *wiring*."""

from .base import Pruner, get_pruner, register_pruner, registered_strategies, validate_keep_fraction
from .metrics import PrunerQualityReport, evaluate_selection

__all__ = [
    "Pruner",
    "PrunerQualityReport",
    "evaluate_selection",
    "get_pruner",
    "register_pruner",
    "registered_strategies",
    "validate_keep_fraction",
]
