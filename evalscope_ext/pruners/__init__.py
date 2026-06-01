"""Pruning strategies. Selection is the *policy*; adapters are the *wiring*."""

from .base import Pruner, get_pruner, register_pruner, registered_strategies, validate_keep_fraction

__all__ = [
    "Pruner",
    "get_pruner",
    "register_pruner",
    "registered_strategies",
    "validate_keep_fraction",
]
