"""Utility functions for image processing and validation."""

import numpy as np
from typing import List, Tuple, Union, Optional
import cv2


def validate_image_stack(image_stack: np.ndarray) -> None:
    """Validate that the input is a proper 3D image stack.

    Args:
        image_stack: Input array to validate

    Raises:
        ValueError: If the array is not a valid 3D image stack
    """
    if not isinstance(image_stack, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if image_stack.ndim != 3:
        raise ValueError(
            f"Expected 3D array (slices, height, width), got {image_stack.ndim}D"
        )

    if image_stack.size == 0:
        raise ValueError("Empty image stack")

    if image_stack.shape[0] < 1:
        raise ValueError("Image stack must have at least 1 slice")


def validate_bbox(
    bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]
) -> None:
    """Validate bounding box coordinates against image dimensions.

    Args:
        bbox: Bounding box (x, y, width, height)
        image_shape: Image shape (height, width)

    Raises:
        ValueError: If bounding box is invalid
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape

    if w <= 0 or h <= 0:
        raise ValueError(f"Bounding box dimensions must be positive, got {w}x{h}")

    if x < 0 or y < 0:
        raise ValueError(
            f"Bounding box coordinates must be non-negative, got ({x}, {y})"
        )

    if x + w > img_w or y + h > img_h:
        raise ValueError(
            f"Bounding box ({x}, {y}, {w}, {h}) exceeds image dimensions ({img_h}, {img_w})"
        )


def normalize_image(image: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Normalize image to [0, 1] range and convert to specified dtype.

    Args:
        image: Input image
        dtype: Target data type

    Returns:
        Normalized image
    """
    if image.size == 0:
        raise ValueError("Cannot normalize empty image")

    # Convert to target dtype
    if image.dtype != dtype:
        image = image.astype(dtype)

    # Normalize to [0, 1] if not already
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:  # Avoid division by zero for constant images
        if img_max > 1.0 or img_min < 0.0:
            image = (image - img_min) / (img_max - img_min)

    return image


def extract_template(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract template region from image using bounding box.

    Args:
        image: Source image
        bbox: Bounding box (x, y, width, height)

    Returns:
        Extracted template region

    Raises:
        ValueError: If bounding box is invalid
    """
    validate_bbox(bbox, image.shape[:2])
    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy()


def create_test_image_stack(
    n_slices: int = 10,
    height: int = 256,
    width: int = 256,
    noise_level: float = 0.1,
    translation_range: float = 5.0,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Create a synthetic image stack for testing with known translations.

    Args:
        n_slices: Number of slices in the stack
        height: Image height
        width: Image width
        noise_level: Amount of Gaussian noise to add (0-1)
        translation_range: Maximum translation in pixels
        dtype: Data type for the images

    Returns:
        Tuple of (image_stack, true_displacements)
    """
    # Create base pattern - a simple geometric shape
    base_image = np.zeros((height, width), dtype=dtype)

    # Add some geometric features
    center_x, center_y = width // 2, height // 2
    cv2.circle(base_image, (center_x, center_y), min(width, height) // 8, 1.0, -1)
    cv2.rectangle(
        base_image,
        (center_x - width // 6, center_y - height // 6),
        (center_x + width // 6, center_y + height // 6),
        0.7,
        2,
    )

    # Create stack with translations
    image_stack = np.zeros((n_slices, height, width), dtype=dtype)
    true_displacements = []

    np.random.seed(42)  # For reproducible results

    for i in range(n_slices):
        # Random translation
        dx = np.random.uniform(-translation_range, translation_range)
        dy = np.random.uniform(-translation_range, translation_range)
        true_displacements.append((dx, dy))

        # Apply translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(
            base_image, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )

        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, translated.shape).astype(dtype)
            translated = np.clip(translated + noise, 0, 1)

        image_stack[i] = translated

    return image_stack, true_displacements


def calculate_alignment_quality(
    displacements: List[Tuple[float, float]],
    reference_displacements: Optional[List[Tuple[float, float]]] = None,
) -> dict:
    """Calculate quality metrics for stack alignment.

    Args:
        displacements: Calculated displacements
        reference_displacements: Known true displacements (for synthetic data)

    Returns:
        Dictionary with quality metrics
    """
    displacements_array = np.array(displacements)

    metrics = {
        "mean_displacement": np.mean(np.sqrt(np.sum(displacements_array**2, axis=1))),
        "std_displacement": np.std(np.sqrt(np.sum(displacements_array**2, axis=1))),
        "max_displacement": np.max(np.sqrt(np.sum(displacements_array**2, axis=1))),
        "dx_range": np.ptp(displacements_array[:, 0]),
        "dy_range": np.ptp(displacements_array[:, 1]),
    }

    if reference_displacements is not None:
        ref_array = np.array(reference_displacements)
        errors = displacements_array - ref_array
        error_magnitudes = np.sqrt(np.sum(errors**2, axis=1))

        metrics.update(
            {
                "mean_error": np.mean(error_magnitudes),
                "std_error": np.std(error_magnitudes),
                "max_error": np.max(error_magnitudes),
                "rmse": np.sqrt(np.mean(error_magnitudes**2)),
            }
        )

    return metrics
