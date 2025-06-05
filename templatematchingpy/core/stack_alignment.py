"""Stack alignment functionality for image registration."""

import cv2
import numpy as np
import warnings
from typing import Tuple, List, Optional, Union

from .config import AlignmentConfig
from .template_matching import TemplateMatchingEngine


class StackAligner:
    """Main stack alignment class for registering image sequences.

    This class provides functionality to align image stacks using template
    matching with configurable precision and search parameters.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None):
        """Initialize the stack aligner.

        Args:
            config: Alignment configuration. If None, uses default settings.
        """
        self.config = config or AlignmentConfig()
        self.matcher = TemplateMatchingEngine()
        
        # Store alignment results
        self.displacements: List[Tuple[float, float]] = []
        self.translation_matrices: Optional[np.ndarray] = None  # Shape: [nframes, 3, 3]
        self.is_registered: bool = False

    def align_slice(
        self, source: np.ndarray, template: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Align a single image slice to a template.

        Args:
            source: Source image to align
            template: Reference template
            bbox: Bounding box (x, y, width, height) defining template region

        Returns:
            (dx, dy) displacement needed to align source to template

        Raises:
            ValueError: If bbox is invalid or template/source have incompatible sizes
        """
        x, y, w, h = bbox

        # Validate bounding box
        if (
            x < 0
            or y < 0
            or w <= 0
            or h <= 0
            or x + w > source.shape[1]
            or y + h > source.shape[0]
        ):
            raise ValueError(
                f"Invalid bounding box {bbox} for image shape {source.shape}"
            )

        # Define search region
        if self.config.search_area > 0:
            search_x = max(0, x - self.config.search_area)
            search_y = max(0, y - self.config.search_area)
            search_w = min(source.shape[1] - search_x, w + 2 * self.config.search_area)
            search_h = min(source.shape[0] - search_y, h + 2 * self.config.search_area)

            search_region = source[
                search_y : search_y + search_h, search_x : search_x + search_w
            ]
            offset_x, offset_y = search_x, search_y

            # Validate template size against search region (not full source)
            if (
                template.shape[0] >= search_region.shape[0]
                or template.shape[1] >= search_region.shape[1]
            ):
                raise ValueError(
                    f"Template size {template.shape} is too large for search region "
                    f"{search_region.shape}. When using search_area={self.config.search_area}, "
                    f"template must be smaller than bbox + 2*search_area. "
                    f"Current search region: {search_region.shape}, "
                    f"bbox: {bbox}, search_area: {self.config.search_area}. "
                    f"Consider: 1) Using smaller template, 2) Increasing search_area, "
                    f"or 3) Setting search_area=0 to search entire image."
                )

            # Informative warning for borderline cases
            if (
                template.shape[0] > search_region.shape[0] * 0.8
                or template.shape[1] > search_region.shape[1] * 0.8
            ):
                warnings.warn(
                    f"Template size {template.shape} is close to search region size "
                    f"{search_region.shape}. This may reduce alignment accuracy. "
                    f"Consider using a smaller template or larger search_area.",
                    UserWarning,
                )
        else:
            search_region = source
            offset_x, offset_y = 0, 0

            # Standard validation for full source matching
            if (
                template.shape[0] >= source.shape[0]
                or template.shape[1] >= source.shape[1]
            ):
                raise ValueError(
                    f"Template size {template.shape} must be smaller than source image "
                    f"size {source.shape} for template matching."
                )

        # Perform template matching
        correlation_map = self.matcher.match_template(
            search_region, template, self.config.method
        )
        peak_x, peak_y = self.matcher.find_peak(correlation_map, self.config.method)

        # Sub-pixel refinement if enabled
        if self.config.subpixel:
            peak_x, peak_y = self.matcher.gaussian_peak_fit(
                correlation_map, peak_x, peak_y
            )

        # Calculate displacement relative to original coordinate space
        if self.config.search_area > 0:
            dx = (offset_x + self.config.search_area) - peak_x
            dy = (offset_y + self.config.search_area) - peak_y
        else:
            dx = x - peak_x
            dy = y - peak_y
        
        return dx, dy

    def _create_translation_matrix(self, dx: float, dy: float) -> np.ndarray:
        """Create 3x3 homogeneous translation matrix.
        
        Args:
            dx: Translation in x direction
            dy: Translation in y direction
            
        Returns:
            3x3 homogeneous transformation matrix
        """
        matrix = np.eye(3, dtype=np.float32)
        matrix[0, 2] = dx
        matrix[1, 2] = dy
        return matrix

    def apply_translation(
        self, 
        image: np.ndarray, 
        matrix: np.ndarray,
        **kwargs: Optional[dict]
    ) -> np.ndarray:
        """Apply translation to an image using affine transformation.

        Args:
            image: Image to be translated
            matrix: 3x3 transformation matrix
            kwargs: Additional parameters for cv2.warpAffine

        Returns:
            Translated image with same dimensions as input
        """
        h, w = image.shape[:2]

        # Convert 3x3 to 2x3 for cv2.warpAffine
        M = matrix[:2, :]
        # Set default borderMode if not provided in kwargs
        if 'borderMode' not in kwargs:
            kwargs['borderMode'] = cv2.BORDER_CONSTANT

        # Apply translation with border replication to handle edge regions
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=self.config.interpolation,
            **kwargs,
        )

    def get_alignment(self, data_type: str) -> Union[List[Tuple[float, float]], np.ndarray]:
        """Retrieve stored displacements and translation matrices.
        
        Args:
            data_type: Type of data to retrieve ('alignment' or 'translation_mat')
        
        Returns:
            Either:
                - displacements: List of (dx, dy) for each slice (if data_type='alignment')
                - translation_matrices: numpy array of shape [nframes, 3, 3] (if data_type='translation_mat')
                
        Raises:
            RuntimeError: If no registration has been performed
            ValueError: If data_type is not 'alignment' or 'translation_mat'
        """
        if not self.is_registered:
            raise RuntimeError("No registration has been performed. Call register_stack() first.")
        
        if data_type not in ['alignment', 'translation_mat']:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be 'alignment' or 'translation_mat'.")
        
        if data_type == 'alignment':
            return self.displacements.copy()
        else:  # data_type == 'translation_mat'
            return self.translation_matrices.copy()

    def register_stack(
        self,
        image_stack: np.ndarray,
        bbox: Tuple[int, int, int, int],
        reference_slice: int = 0
    ) -> np.ndarray:
        """Register image stack and store alignment parameters.
        
        Args:
            image_stack: 3D numpy array (slices, height, width)
            bbox: Template bounding box (x, y, width, height)
            reference_slice: Index of reference slice
            
        Returns:
            aligned_stack: Registered image stack
            
        Side Effects:
            - Stores displacements in self.displacements
            - Stores translation matrices in self.translation_matrices
            - Sets self.is_registered = True
        """
        if image_stack.ndim != 3:
            raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")

        n_slices = image_stack.shape[0]

        if not (0 <= reference_slice < n_slices):
            raise IndexError(
                f"Reference slice {reference_slice} out of bounds [0, {n_slices - 1}]"
            )

        x, y, w, h = bbox

        # Validate bounding box against image dimensions
        if y + h > image_stack.shape[1] or x + w > image_stack.shape[2]:
            raise ValueError(
                f"Bounding box {bbox} exceeds image dimensions {image_stack.shape[1:]}"
            )

        # Clear previous registration data and initialize arrays
        self.displacements = []
        self.translation_matrices = np.zeros((n_slices, 3, 3), dtype=np.float32)
        
        aligned_stack = image_stack.copy()

        # Extract reference template
        reference = image_stack[reference_slice, y : y + h, x : x + w]

        for i in range(n_slices):
            if i == reference_slice:
                # Reference slice has no displacement
                self.displacements.append((0.0, 0.0))
                self.translation_matrices[i] = self._create_translation_matrix(0.0, 0.0)
                continue

            # Calculate displacement for this slice
            dx, dy = self.align_slice(image_stack[i], reference, bbox)
            self.displacements.append((dx, dy))
            
            # Create and store translation matrix
            matrix = self._create_translation_matrix(dx, dy)
            self.translation_matrices[i] = matrix

            # Apply translation to align the slice
            aligned_stack[i] = self.apply_translation(image_stack[i], matrix=matrix)

        self.is_registered = True
        return aligned_stack

    def transform_stack(self, image_stack: np.ndarray) -> np.ndarray:
        """Apply stored translation matrices to a new image stack.
        
        Args:
            image_stack: 3D numpy array to be transformed
            
        Returns:
            transformed_stack: Stack with stored transformations applied
            
        Raises:
            RuntimeError: If no registration has been performed
            ValueError: If stack dimensions don't match stored parameters
        """
        if not self.is_registered:
            raise RuntimeError("No registration has been performed. Call register_stack() first.")
        
        if image_stack.ndim != 3:
            raise ValueError(f"Expected 3D image stack, got {image_stack.ndim}D")
        
        if image_stack.shape[0] != self.translation_matrices.shape[0]:
            raise ValueError(
                f"Stack has {image_stack.shape[0]} slices but {self.translation_matrices.shape[0]} "
                f"transformation matrices stored. Stack dimensions must match registration."
            )
        
        transformed_stack = image_stack.copy()
        
        for i in range(self.translation_matrices.shape[0]):
            matrix = self.translation_matrices[i]
            transformed_stack[i] = self.apply_translation(image_stack[i], matrix=matrix)
        
        return transformed_stack


def register_stack(
    image_stack: np.ndarray,
    bbox: Tuple[int, int, int, int],
    reference_slice: int = 0,
    config: Optional[AlignmentConfig] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Register an image stack using template matching.

    This is the main function for aligning image stacks. It registers all slices
    to a reference slice using template matching within a specified region.

    Args:
        image_stack: 3D numpy array (slices, height, width)
        bbox: Template bounding box (x, y, width, height)
        reference_slice: Index of the reference slice (default: 0)
        config: Alignment configuration (default: AlignmentConfig())

    Returns:
        Tuple containing:
            - aligned_stack: 3D array of aligned images
            - displacements: List of (dx, dy) displacements for each slice

    Raises:
        ValueError: If inputs are invalid
        IndexError: If reference_slice is out of bounds
    """
    aligner = StackAligner(config)
    aligned_stack = aligner.register_stack(image_stack, bbox, reference_slice)
    displacements = aligner.get_alignment('alignment')
    return aligned_stack, displacements
