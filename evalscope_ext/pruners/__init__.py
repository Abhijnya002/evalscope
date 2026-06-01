"""Pruning strategies. Selection is the *policy*; adapters are the *wiring*."""

from .base import Pruner, get_pruner, register_pruner, registered_strategies, validate_keep_fraction
from .judge_noise_aware import JudgeNoiseAwarePruner
from .metrics import PrunerQualityReport, evaluate_selection
from .mmmu_encoder_probe import MMMUEncoderProbePruner, ProbePair
from .stratified import StratifiedDiscriminativePruner

__all__ = [
    "JudgeNoiseAwarePruner",
    "MMMUEncoderProbePruner",
    "ProbePair",
    "Pruner",
    "PrunerQualityReport",
    "StratifiedDiscriminativePruner",
    "evaluate_selection",
    "get_pruner",
    "register_pruner",
    "registered_strategies",
    "validate_keep_fraction",
]
