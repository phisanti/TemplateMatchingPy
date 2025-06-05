"""TemplateMatchingPy: Python implementation of ImageJ template matching and stack alignment.

This package provides programmatic, GUI-free interface for:
- Template matching using OpenCV's correlation methods
- Image stack alignment with sub-pixel precision
- Batch processing capabilities for microscopy workflows

Original ImageJ plugin by Qingzong Tseng.
"""

from .core import AlignmentConfig, TemplateMatchingEngine, StackAligner, register_stack

from .utils import (
    validate_image_stack,
    normalize_image,
    create_test_image_stack,
    calculate_alignment_quality,
)

__version__ = "1.0.0"
__author__ = "TemplateMatchingPy Contributors"
__email__ = "contact@example.com"

__all__ = [
    # Main API
    "register_stack",
    "AlignmentConfig",
    # Core classes
    "TemplateMatchingEngine",
    "StackAligner",
    # Utility functions
    "validate_image_stack",
    "normalize_image",
    "create_test_image_stack",
    "calculate_alignment_quality",
]
