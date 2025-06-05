"""Configuration and constants for TemplateMatchingPy."""

import cv2
from dataclasses import dataclass
from typing import Optional


# Small constant to avoid division by zero or log of zero
EPS = 1e-10


@dataclass
class AlignmentConfig:
    """Configuration for template matching and stack alignment.

    Attributes:
        method: OpenCV template matching method (0-5)
            0: TM_SQDIFF
            1: TM_SQDIFF_NORMED
            2: TM_CCORR
            3: TM_CCORR_NORMED
            4: TM_CCOEFF
            5: TM_CCOEFF_NORMED (default)
        search_area: Pixels around ROI to search for matches (0 = entire image)
        subpixel: Enable sub-pixel registration using Gaussian fitting
        interpolation: OpenCV interpolation method for image warping
    """

    method: int = 5  # TM_CCOEFF_NORMED
    search_area: int = 0
    subpixel: bool = True
    interpolation: int = cv2.INTER_LINEAR

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.method not in range(6):
            raise ValueError(f"Method must be 0-5, got {self.method}")
        if self.search_area < 0:
            raise ValueError(f"Search area must be >= 0, got {self.search_area}")


# Mapping of method indices to OpenCV constants
MATCHING_METHODS = {
    0: cv2.TM_SQDIFF,
    1: cv2.TM_SQDIFF_NORMED,
    2: cv2.TM_CCORR,
    3: cv2.TM_CCORR_NORMED,
    4: cv2.TM_CCOEFF,
    5: cv2.TM_CCOEFF_NORMED,
}

# Methods that expect minimum value (others expect maximum)
MIN_VALUE_METHODS = {0, 1}
