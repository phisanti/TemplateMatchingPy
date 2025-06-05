"""Tests for template matching functionality."""

import unittest
import numpy as np
import cv2
from templatematchingpy.core.template_matching import TemplateMatchingEngine
from templatematchingpy.core.config import EPS


class TestTemplateMatchingEngine(unittest.TestCase):
    """Test cases for TemplateMatchingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = TemplateMatchingEngine()

        # Create simple test images
        self.source = np.ones((100, 100), dtype=np.float32)
        self.template = np.ones((20, 20), dtype=np.float32) * 0.5

        # Create a more complex test case with known peak
        self.complex_source = np.zeros((100, 100), dtype=np.float32)
        self.complex_template = np.ones((10, 10), dtype=np.float32)
        # Place template at known location (30, 40)
        self.complex_source[40:50, 30:40] = 1.0

    def test_init(self):
        """Test engine initialization."""
        self.assertEqual(len(self.engine.supported_methods), 6)
        self.assertIn(0, self.engine.supported_methods)
        self.assertIn(5, self.engine.supported_methods)

    def test_convert_and_normalize_float32(self):
        """Test image conversion for float32 images."""
        img = np.array([[0.5, 1.0], [0.0, 0.8]], dtype=np.float32)
        result = self.engine._convert_and_normalize(img)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, img)

    def test_convert_and_normalize_uint16(self):
        """Test image conversion for 16-bit images."""
        img = np.array([[32767, 65535], [0, 40000]], dtype=np.uint16)
        result = self.engine._convert_and_normalize(img)

        self.assertEqual(result.dtype, np.float32)
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

    def test_convert_and_normalize_uint8(self):
        """Test image conversion for 8-bit images."""
        img = np.array([[127, 255], [0, 200]], dtype=np.uint8)
        result = self.engine._convert_and_normalize(img)

        self.assertEqual(result.dtype, np.float32)
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

    def test_match_template_basic(self):
        """Test basic template matching."""
        for method in range(6):
            with self.subTest(method=method):
                result = self.engine.match_template(self.source, self.template, method)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.dtype, np.float32)
                expected_shape = (81, 81)  # 100-20+1, 100-20+1
                self.assertEqual(result.shape, expected_shape)

    def test_match_template_invalid_method(self):
        """Test template matching with invalid method."""
        with self.assertRaises(ValueError):
            self.engine.match_template(self.source, self.template, 6)

        with self.assertRaises(ValueError):
            self.engine.match_template(self.source, self.template, -1)

    def test_match_template_template_too_large(self):
        """Test template matching when template is larger than source."""
        large_template = np.ones((200, 200), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.engine.match_template(self.source, large_template, 5)

    def test_find_peak_max_methods(self):
        """Test peak finding for methods that expect maximum."""
        # Create correlation map with known peak
        correlation_map = np.zeros((10, 10), dtype=np.float32)
        correlation_map[3, 7] = 1.0  # Peak at (7, 3)

        for method in [2, 3, 4, 5]:  # Methods that expect maximum
            with self.subTest(method=method):
                x, y = self.engine.find_peak(correlation_map, method)
                self.assertEqual((x, y), (7, 3))

    def test_find_peak_min_methods(self):
        """Test peak finding for methods that expect minimum."""
        # Create correlation map with known minimum
        correlation_map = np.ones((10, 10), dtype=np.float32)
        correlation_map[3, 7] = 0.0  # Minimum at (7, 3)

        for method in [0, 1]:  # Methods that expect minimum
            with self.subTest(method=method):
                x, y = self.engine.find_peak(correlation_map, method)
                self.assertEqual((x, y), (7, 3))

    def test_gaussian_peak_fit_center(self):
        """Test Gaussian peak fitting at image center."""
        # Create a simple peak in correlation map
        correlation_map = np.zeros((10, 10), dtype=np.float32)
        correlation_map[5, 5] = 1.0
        correlation_map[4:7, 4:7] = 0.5  # Neighbors

        x_refined, y_refined = self.engine.gaussian_peak_fit(correlation_map, 5, 5)

        # Should be close to integer coordinates for symmetric peak
        self.assertAlmostEqual(x_refined, 5.0, places=1)
        self.assertAlmostEqual(y_refined, 5.0, places=1)

    def test_gaussian_peak_fit_boundary(self):
        """Test Gaussian peak fitting at image boundaries."""
        correlation_map = np.zeros((10, 10), dtype=np.float32)
        correlation_map[0, 0] = 1.0

        # Should return original coordinates at boundary
        x_refined, y_refined = self.engine.gaussian_peak_fit(correlation_map, 0, 0)
        self.assertEqual(x_refined, 0.0)
        self.assertEqual(y_refined, 0.0)

    def test_gaussian_peak_fit_asymmetric(self):
        """Test Gaussian peak fitting with asymmetric peak."""
        correlation_map = np.zeros((10, 10), dtype=np.float32)
        correlation_map[5, 5] = 1.0
        correlation_map[5, 4] = 0.8  # Stronger on left
        correlation_map[5, 6] = 0.3  # Weaker on right
        correlation_map[4, 5] = 0.6  # Moderate above
        correlation_map[6, 5] = 0.4  # Weaker below

        x_refined, y_refined = self.engine.gaussian_peak_fit(correlation_map, 5, 5)

        # Peak should shift towards stronger neighbors
        self.assertLess(x_refined, 5.0)  # Should shift left
        self.assertLess(y_refined, 5.0)  # Should shift up

    def test_gaussian_peak_fit_invalid_values(self):
        """Test Gaussian peak fitting with problematic values."""
        # Test with zero/negative values
        correlation_map = np.zeros((10, 10), dtype=np.float32)
        correlation_map[5, 5] = 0.0

        x_refined, y_refined = self.engine.gaussian_peak_fit(correlation_map, 5, 5)
        self.assertEqual(x_refined, 5.0)
        self.assertEqual(y_refined, 5.0)

    def test_integration_subpixel_refinement(self):
        """Integration test with sub-pixel refinement."""
        correlation_map = self.engine.match_template(
            self.complex_source, self.complex_template, 5
        )
        x_int, y_int = self.engine.find_peak(correlation_map, 5)
        x_sub, y_sub = self.engine.gaussian_peak_fit(correlation_map, x_int, y_int)

        # Sub-pixel coordinates should be close to integer coordinates
        self.assertAlmostEqual(x_sub, x_int, delta=0.5)
        self.assertAlmostEqual(y_sub, y_int, delta=0.5)

    def test_edge_cases_empty_arrays(self):
        """Test edge cases with empty arrays."""
        empty_source = np.array([], dtype=np.float32).reshape(0, 0)
        empty_template = np.array([], dtype=np.float32).reshape(0, 0)

        with self.assertRaises((ValueError, cv2.error)):
            self.engine.match_template(empty_source, empty_template, 5)

    def test_edge_cases_single_pixel(self):
        """Test edge cases with single pixel images."""
        single_source = np.array([[1.0]], dtype=np.float32)
        single_template = np.array([[1.0]], dtype=np.float32)

        result = self.engine.match_template(single_source, single_template, 5)
        self.assertEqual(result.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
