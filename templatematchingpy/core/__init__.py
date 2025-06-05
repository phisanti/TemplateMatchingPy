"""Core functionality for TemplateMatchingPy."""

from .config import AlignmentConfig, MATCHING_METHODS, MIN_VALUE_METHODS, EPS
from .template_matching import TemplateMatchingEngine
from .stack_alignment import StackAligner, register_stack

__all__ = [
    "AlignmentConfig",
    "TemplateMatchingEngine",
    "StackAligner",
    "register_stack",
    "MATCHING_METHODS",
    "MIN_VALUE_METHODS",
    "EPS",
]
