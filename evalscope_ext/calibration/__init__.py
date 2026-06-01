"""Calibration data loaders.

The shipped ``Evals/`` directory contains per-sample reviews and
predictions for three reference models on LCB v5 and AA-LCR (and one
model on MMMU). The pruners consume these as the calibration set used
to characterise the benchmark population — *not* to overfit to the
reference models.
"""

from .loader import (
    CalibrationBundle,
    DEFAULT_CALIBRATION_ROOT,
    SampleScores,
    difficulty_bins,
    load_calibration,
)

__all__ = [
    "CalibrationBundle",
    "DEFAULT_CALIBRATION_ROOT",
    "SampleScores",
    "difficulty_bins",
    "load_calibration",
]
