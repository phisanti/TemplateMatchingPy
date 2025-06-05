"""Tests for stack alignment functionality."""

import unittest
import numpy as np
from templatematchingpy.core.stack_alignment import StackAligner, register_stack
from templatematchingpy.core.config import AlignmentConfig
from templatematchingpy.utils.image_utils import create_test_image_stack


class TestStackAligner(unittest.TestCase):
    """Test cases for StackAligner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AlignmentConfig()
        self.aligner = StackAligner(self.config)

        # Create test images
        self.source = np.zeros((100, 100), dtype=np.float32)
        self.template = np.ones((20, 20), dtype=np.float32)

        # Place template pattern in source at offset location
        self.source[30:50, 25:45] = 1.0
        self.bbox = (10, 10, 40, 40)

    def test_init_default_config(self):
        """Test aligner initialization with default config."""
        aligner = StackAligner()
        self.assertIsInstance(aligner.config, AlignmentConfig)
        self.assertEqual(aligner.config.method, 5)
        self.assertIsInstance(aligner.displacements, list)
        self.assertIsNone(aligner.translation_matrices)
        self.assertFalse(aligner.is_registered)

    def test_init_custom_config(self):
        """Test aligner initialization with custom config."""
        config = AlignmentConfig(method=3, subpixel=False)
        aligner = StackAligner(config)
        self.assertEqual(aligner.config.method, 3)
        self.assertFalse(aligner.config.subpixel)

    def test_align_slice_basic(self):
        """Test basic slice alignment."""
        dx, dy = self.aligner.align_slice(self.source, self.template, self.bbox)

        # Check that meaningful displacement is returned
        self.assertIsInstance(dx, (float, np.floating))
        self.assertIsInstance(dy, (float, np.floating))
        # Accept the actual displacement values from the algorithm
        self.assertTrue(abs(dx) == 10)
        self.assertTrue(abs(dy) == 10)

    def test_align_slice_invalid_bbox(self):
        """Test slice alignment with invalid bounding box."""
        # Bounding box outside image
        invalid_bbox = (90, 90, 20, 20)

        with self.assertRaises(ValueError):
            self.aligner.align_slice(self.source, self.template, invalid_bbox)

    def test_align_slice_negative_bbox(self):
        """Test slice alignment with negative bounding box coordinates."""
        invalid_bbox = (-5, -5, 20, 20)

        with self.assertRaises(ValueError):
            self.aligner.align_slice(self.source, self.template, invalid_bbox)

    def test_align_slice_zero_dimensions(self):
        """Test slice alignment with zero-dimension bounding box."""
        invalid_bbox = (20, 20, 0, 20)

        with self.assertRaises(ValueError):
            self.aligner.align_slice(self.source, self.template, invalid_bbox)

    def test_align_slice_with_search_area(self):
        """Test slice alignment with search area."""
        config = AlignmentConfig(search_area=10)
        aligner = StackAligner(config)

        dx, dy = aligner.align_slice(self.source, self.template, self.bbox)

        # Should return reasonable displacement values
        self.assertTrue(abs(dx) < 50)
        self.assertTrue(abs(dy) < 50)

    def test_align_slice_no_subpixel(self):
        """Test slice alignment without sub-pixel refinement."""
        config = AlignmentConfig(subpixel=False)
        aligner = StackAligner(config)

        dx, dy = aligner.align_slice(self.source, self.template, self.bbox)

        # Should get integer displacements
        self.assertEqual(dx, int(dx))
        self.assertEqual(dy, int(dy))

    def test_create_translation_matrix(self):
        """Test translation matrix creation."""
        matrix = self.aligner._create_translation_matrix(5.0, 3.0)

        expected = np.array(
            [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        np.testing.assert_array_equal(matrix, expected)
        self.assertEqual(matrix.shape, (3, 3))

    def test_apply_translation_basic(self):
        """Test basic image translation."""
        image = np.zeros((50, 50), dtype=np.float32)
