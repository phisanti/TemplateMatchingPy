"""Core template matching functionality using OpenCV."""

import cv2
import numpy as np
from typing import Tuple

from .config import MATCHING_METHODS, MIN_VALUE_METHODS, EPS


class TemplateMatchingEngine:
    """Core template matching functionality using OpenCV correlation methods.

    This class provides methods for template matching with sub-pixel precision,
    supporting all OpenCV template matching methods.
    """

    def __init__(self):
        """Initialize the template matching engine."""
        self.supported_methods = MATCHING_METHODS.copy()

    def _convert_and_normalize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to float32 and normalize if needed.

        Args:
            image: Input image array

        Returns:
            Normalized float32 image
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize if image's maximum is greater than 1.0 (typically for 16-bit images)
        if image.max() > 1.0:
            max_val = image.max() + EPS
            image = image / max_val
        return image

    def match_template(
        self, source: np.ndarray, template: np.ndarray, method: int
    ) -> np.ndarray:
        """Perform template matching using OpenCV Backend.

        Args:
            source: Source image array
            template: Template image array
            method: Matching method (0-5)

        Returns:
            Correlation map as a float32 array

        Raises:
            ValueError: If method is not supported
            ValueError: If template is larger than source
        """
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {method}")

        if template.shape[0] > source.shape[0] or template.shape[1] > source.shape[1]:
            raise ValueError("Template cannot be larger than source image")

        source = self._convert_and_normalize(source)
        template = self._convert_and_normalize(template)

        cv_method = self.supported_methods[method]
        result = cv2.matchTemplate(source, template, cv_method)
        return result

    def find_peak(self, correlation_map: np.ndarray, method: int) -> Tuple[int, int]:
        """Find the peak location in a correlation map.

        Args:
            correlation_map: Correlation map from template matching
            method: Matching method used (affects whether to find min or max)

        Returns:
            (x, y) location of the peak
        """
        if method in MIN_VALUE_METHODS:  # Square difference methods expect minimum
            _, _, min_loc, _ = cv2.minMaxLoc(correlation_map)
            return min_loc
        else:
            _, _, _, max_loc = cv2.minMaxLoc(correlation_map)
            return max_loc

    def gaussian_peak_fit(
        self, correlation_map: np.ndarray, x: int, y: int
    ) -> Tuple[float, float]:
        """Perform sub-pixel peak refinement using Gaussian fitting.

        Uses logarithmic Gaussian fitting to estimate sub-pixel peak location
        based on the peak and its immediate neighbors.

        Args:
            correlation_map: The correlation map
            x: Peak x-coordinate in integer coordinates
            y: Peak y-coordinate in integer coordinates

        Returns:
            Refined (x, y) peak as float values
        """
        h, w = correlation_map.shape

        # Check boundaries - need at least 1 pixel margin for fitting
        if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
            return float(x), float(y)

        try:
            # Retrieve neighborhood values with epsilon to avoid log(0)
            center = correlation_map[y, x] + EPS
            left = correlation_map[y, x - 1] + EPS
            right = correlation_map[y, x + 1] + EPS
            up = correlation_map[y - 1, x] + EPS
            down = correlation_map[y + 1, x] + EPS

            # Calculate logarithms
            log_center = np.log(center)
            log_left = np.log(left)
            log_right = np.log(right)
            log_up = np.log(up)
            log_down = np.log(down)

            # Calculate sub-pixel offsets using parabolic fitting
            dx_num = log_left - log_right
            dx_den = 2 * log_left - 4 * log_center + 2 * log_right
            dy_num = log_up - log_down
            dy_den = 2 * log_up - 4 * log_center + 2 * log_down

            # Guard against division by zero
            if abs(dx_den) < EPS or abs(dy_den) < EPS:
                return float(x), float(y)

            dx = dx_num / dx_den
            dy = dy_num / dy_den

            # Limit sub-pixel offsets to reasonable range
            dx = np.clip(dx, -1.0, 1.0)
            dy = np.clip(dy, -1.0, 1.0)

            return x + dx, y + dy

        except (ValueError, RuntimeWarning):
            # Return original coordinates if fitting fails
            return float(x), float(y)
