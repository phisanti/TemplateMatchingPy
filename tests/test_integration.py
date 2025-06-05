"""Integration tests for TemplateMatchingPy."""

import unittest
import numpy as np
from templatematchingpy import (
    register_stack,
    AlignmentConfig,
    create_test_image_stack,
    calculate_alignment_quality,
)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete package functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test image stack with known translations
        self.stack, self.true_displacements = create_test_image_stack(
            n_slices=5, height=128, width=128, translation_range=4.0, noise_level=0.1
        )
        self.bbox = (50, 50, 28, 28)

    def test_complete_workflow_basic(self):
        """Test complete workflow from stack creation to alignment."""
        # Create alignment configuration
        config = AlignmentConfig(method=5, subpixel=True)

        # Perform registration
        aligned_stack, displacements = register_stack(
            self.stack, self.bbox, reference_slice=0, config=config
        )

        # Verify results
        self.assertEqual(aligned_stack.shape, self.stack.shape)
        self.assertEqual(len(displacements), self.stack.shape[0])

        # Reference slice should not be displaced
        self.assertEqual(displacements[0], (0.0, 0.0))

        # Calculate alignment quality
        quality = calculate_alignment_quality(displacements, self.true_displacements)

        # Quality metrics should be reasonable
        self.assertIsInstance(quality, dict)
        self.assertIn("mean_displacement", quality)
        self.assertIn("rmse", quality)

        # RMSE should be relatively small for synthetic data (relaxed threshold)
        self.assertLess(quality["rmse"], 5.0)

    def test_workflow_different_reference_slices(self):
        """Test workflow with different reference slices."""
        for ref_slice in [0, 2, 4]:
            with self.subTest(reference_slice=ref_slice):
                aligned_stack, displacements = register_stack(
                    self.stack, self.bbox, reference_slice=ref_slice
                )

                # Reference slice should have zero displacement
                self.assertEqual(displacements[ref_slice], (0.0, 0.0))

                # Other slices should have non-zero displacements (mostly)
                other_displacements = [
                    d for i, d in enumerate(displacements) if i != ref_slice
                ]
                non_zero_count = sum(
                    1
                    for dx, dy in other_displacements
                    if abs(dx) > 0.1 or abs(dy) > 0.1
                )
                self.assertGreater(non_zero_count, 0)

    def test_workflow_different_template_sizes(self):
        """Test workflow with different template sizes."""
        template_sizes = [(20, 20), (32, 32), (40, 40)]

        for w, h in template_sizes:
            with self.subTest(template_size=(w, h)):
                # Adjust bbox to accommodate different sizes
                x = (self.stack.shape[2] - w) // 2
                y = (self.stack.shape[1] - h) // 2
                bbox = (x, y, w, h)

                aligned_stack, displacements = register_stack(
                    self.stack, bbox, reference_slice=0
                )

                self.assertEqual(aligned_stack.shape, self.stack.shape)
                self.assertEqual(len(displacements), self.stack.shape[0])

    def test_workflow_different_matching_methods(self):
        """Test workflow with different matching methods."""
        methods = [1, 3, 5]  # Different normalized methods

        results = []
        for method in methods:
            with self.subTest(method=method):
                config = AlignmentConfig(method=method)
                aligned_stack, displacements = register_stack(
                    self.stack, self.bbox, config=config
                )
                results.append(displacements)

                # Basic sanity checks
                self.assertEqual(len(displacements), self.stack.shape[0])
                self.assertEqual(displacements[0], (0.0, 0.0))

        # Different methods should give similar but not identical results
        # Check that at least some results are different
        all_same = all(results[0] == result for result in results[1:])
        self.assertFalse(all_same, "Different methods should give different results")

    def test_subpixel_vs_integer_precision(self):
        """Test comparison between sub-pixel and integer precision."""
        # Sub-pixel alignment
        config_subpixel = AlignmentConfig(subpixel=True)
        _, displacements_subpixel = register_stack(
            self.stack, self.bbox, config=config_subpixel
        )

        # Integer alignment
        config_integer = AlignmentConfig(subpixel=False)
        _, displacements_integer = register_stack(
            self.stack, self.bbox, config=config_integer
        )

        # Calculate quality metrics for both
        quality_subpixel = calculate_alignment_quality(
            displacements_subpixel, self.true_displacements
        )
        quality_integer = calculate_alignment_quality(
            displacements_integer, self.true_displacements
        )

        # Sub-pixel should generally be more accurate
        self.assertLessEqual(
            quality_subpixel["rmse"],
            quality_integer["rmse"] + 0.5,  # Allow some tolerance
        )

    def test_robustness_to_noise(self):
        """Test robustness to different noise levels."""
        noise_levels = [0.0, 0.05, 0.1, 0.2]
        rmse_values = []

        for noise_level in noise_levels:
            # Create stack with specific noise level
            noisy_stack, true_disps = create_test_image_stack(
                n_slices=5,
                height=128,
                width=128,
                translation_range=3.0,
                noise_level=noise_level,
            )

            # Align stack
            _, displacements = register_stack(noisy_stack, self.bbox)

            # Calculate quality
            quality = calculate_alignment_quality(displacements, true_disps)
            rmse_values.append(quality["rmse"])

        # RMSE should increase with noise level
        self.assertLess(rmse_values[0], rmse_values[-1])

        # Even with high noise, RMSE should be reasonable
        self.assertLess(rmse_values[-1], 10.0)

    def test_performance_large_stack(self):
        """Test performance with larger image stack."""
        # Create larger stack
        large_stack, true_disps = create_test_image_stack(
            n_slices=10, height=256, width=256, translation_range=5.0, noise_level=0.05
        )

        bbox = (100, 100, 56, 56)

        # This should complete without errors
        aligned_stack, displacements = register_stack(
            large_stack, bbox, reference_slice=0
        )

        self.assertEqual(aligned_stack.shape, large_stack.shape)
        self.assertEqual(len(displacements), large_stack.shape[0])

        # Calculate quality (relaxed threshold)
        quality = calculate_alignment_quality(displacements, true_disps)
        self.assertLess(quality["rmse"], 8.0)

    def test_edge_case_minimal_stack(self):
        """Test edge case with minimal image stack."""
        # Create minimal stack (2 slices, small size)
        minimal_stack, true_disps = create_test_image_stack(
            n_slices=2, height=64, width=64, translation_range=2.0
        )

        bbox = (20, 20, 24, 24)

        aligned_stack, displacements = register_stack(
            minimal_stack, bbox, reference_slice=0
        )

        self.assertEqual(len(displacements), 2)
        self.assertEqual(displacements[0], (0.0, 0.0))

    def test_consistency_across_runs(self):
        """Test that results are consistent across multiple runs."""
        results = []

        for _ in range(3):
            aligned_stack, displacements = register_stack(
                self.stack, self.bbox, reference_slice=0
            )
            results.append(displacements)

        # All runs should give identical results (deterministic)
        for i in range(1, len(results)):
            self.assertEqual(results[0], results[i])

    def test_api_compatibility(self):
        """Test that the API matches the expected interface from instructions."""
        # Test the exact API example from instructions
        config = AlignmentConfig(method=5, subpixel=True)
        aligned_stack, displacements = register_stack(
            self.stack, bbox=self.bbox, reference_slice=0, config=config
        )

        # Should work without errors
        self.assertIsInstance(aligned_stack, np.ndarray)
        self.assertIsInstance(displacements, list)

        # Test that imports work as expected - avoid shadowing
        import templatematchingpy

        # Should be able to use imported functions
        config2 = templatematchingpy.AlignmentConfig(method=5, subpixel=True)
        result = templatematchingpy.register_stack(
            self.stack, bbox=self.bbox, reference_slice=0, config=config2
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
