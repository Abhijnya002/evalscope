"""Abstract `Pruner` interface + a registry.

A *pruner* picks a subset of upstream sample indices given the
calibration data. The interface is deliberately tiny — pruners get the
calibration bundle and a target keep-fraction, and return the indices
to keep. They never know about evalscope's adapter machinery; the
adapter does the wiring.

Two reasons for the registry: (1) the rubric's run contract names a
strategy by string in ``--dataset-args '{"pruning_strategy": "..."}'``;
(2) tests and the compare-runs CLI need to instantiate a strategy from
its name without importing the concrete class.
"""

from __future__ import annotations

import abc
from typing import Any, Callable

from ..calibration import CalibrationBundle


class Pruner(abc.ABC):
    """Pick a subset of sample indices from a calibration bundle.

    A pruner is *stateless across runs* — same inputs must give the same
    output. We accept ``rng_seed`` rather than hold internal state.
    """

    #: Strategy name as it appears in --dataset-args. Subclasses override.
    name: str = "<unset>"

    @abc.abstractmethod
    def prune(
        self,
        bundle: CalibrationBundle,
        *,
        keep_fraction: float,
        rng_seed: int = 0,
    ) -> list[int]:
        """Return the upstream sample indices to keep.

        ``keep_fraction`` is in (0, 1]. ``keep_fraction=1.0`` returns
        every index (no-op); pruners must handle that gracefully so the
        adapter can fall back to the full set.
        """

    # Implementations may override these to expose human-readable info.
    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


# --- Registry ----------------------------------------------------------------

_REGISTRY: dict[str, type[Pruner]] = {}


def register_pruner(cls: type[Pruner]) -> type[Pruner]:
    """Class decorator to register a Pruner subclass by its ``name``."""
    if not isinstance(cls.name, str) or cls.name == "<unset>":
        raise ValueError(f"{cls.__name__} must set a non-empty `name` class attribute")
    if cls.name in _REGISTRY:
        raise ValueError(f"pruner name {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_pruner(name: str, **kwargs: Any) -> Pruner:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pruner {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def registered_strategies() -> list[str]:
    return sorted(_REGISTRY)


# --- Validation helper -------------------------------------------------------


def validate_keep_fraction(keep_fraction: float) -> None:
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(
            f"keep_fraction must be in (0, 1]; got {keep_fraction!r}"
        )
