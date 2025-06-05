"""Advanced alignment example for TemplateMatchingPy.

This example demonstrates advanced features including:
- Different template matching methods
- Search area restrictions
- Performance comparison
- Quality assessment
"""

import numpy as np
import time
from templatematchingpy import (
    register_stack,
    AlignmentConfig,
    create_test_image_stack,
    calculate_alignment_quality,
)


def compare_alignment_methods():
    """Compare different template matching methods."""
    print("Comparing Template Matching Methods")
    print("=" * 50)

    # Create test data
    image_stack, true_displacements = create_test_image_stack(
        n_slices=6, height=200, width=200, translation_range=4.0, noise_level=0.15
    )

    bbox = (75, 75, 50, 50)

    # Test different methods
    methods = {
        1: "TM_SQDIFF_NORMED",
        3: "TM_CCORR_NORMED",
        4: "TM_CCOEFF",
        5: "TM_CCOEFF_NORMED",
    }

    results = {}

    for method_id, method_name in methods.items():
        print(f"\nTesting {method_name} (method {method_id})...")

        config = AlignmentConfig(method=method_id, subpixel=True)

        start_time = time.time()
        aligned_stack, displacements = register_stack(
            image_stack, bbox, reference_slice=0, config=config
        )
        processing_time = time.time() - start_time

        # Calculate quality metrics
        quality = calculate_alignment_quality(displacements, true_displacements)

        results[method_name] = {
            "displacements": displacements,
            "quality": quality,
            "time": processing_time,
        }

        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  RMSE: {quality['rmse']:.3f} pixels")
        print(f"  Mean error: {quality['mean_error']:.3f} pixels")

    # Summary comparison
    print("\nMethod Comparison Summary:")
    print("-" * 70)
    print(f"{'Method':<20} {'RMSE':<10} {'Mean Error':<12} {'Time (s)':<10}")
    print("-" * 70)

    for method_name, result in results.items():
        quality = result["quality"]
        print(
            f"{method_name:<20} {quality['rmse']:<10.3f} "
            f"{quality['mean_error']:<12.3f} {result['time']:<10.3f}"
        )

    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]["quality"]["rmse"])
    print(
        f"\nBest method by RMSE: {best_method[0]} "
        f"(RMSE: {best_method[1]['quality']['rmse']:.3f})"
    )

    return results


def demonstrate_search_area_effects():
    """Demonstrate the effect of search area on alignment."""
    print("\n\nDemonstrating Search Area Effects")
    print("=" * 50)

    # Create test data with larger translations
    image_stack, true_displacements = create_test_image_stack(
        n_slices=5,
        height=150,
        width=150,
        translation_range=8.0,  # Larger translations
        noise_level=0.1,
    )

    bbox = (50, 50, 40, 40)

    # Test different search areas
    search_areas = [0, 5, 10, 20, 30]

    print("Search Area | RMSE  | Processing Time | Success")
    print("-" * 50)

    for search_area in search_areas:
        config = AlignmentConfig(method=5, subpixel=True, search_area=search_area)

        try:
            start_time = time.time()
            aligned_stack, displacements = register_stack(
                image_stack, bbox, reference_slice=0, config=config
            )
            processing_time = time.time() - start_time

            quality = calculate_alignment_quality(displacements, true_displacements)
            rmse = quality["rmse"]
            success = "Yes"

        except Exception as e:
            processing_time = 0
            rmse = float("inf")
            success = f"No ({str(e)[:20]}...)"

        search_desc = "Full image" if search_area == 0 else f"{search_area} pixels"
        print(
            f"{search_desc:<11} | {rmse:<5.2f} | {processing_time:<15.3f} | {success}"
        )


def subpixel_precision_analysis():
    """Analyze the effect of sub-pixel precision."""
    print("\n\nSub-pixel Precision Analysis")
    print("=" * 50)

    # Create test data with fractional translations
    image_stack, true_displacements = create_test_image_stack(
        n_slices=8,
        height=180,
        width=180,
        translation_range=3.5,  # Fractional translations likely
        noise_level=0.05,
    )

    bbox = (65, 65, 50, 50)

    # Compare integer vs sub-pixel alignment
    configs = {
        "Integer precision": AlignmentConfig(method=5, subpixel=False),
        "Sub-pixel precision": AlignmentConfig(method=5, subpixel=True),
    }

    for config_name, config in configs.items():
        print(f"\n{config_name}:")

        aligned_stack, displacements = register_stack(
            image_stack, bbox, reference_slice=0, config=config
        )

        quality = calculate_alignment_quality(displacements, true_displacements)

        print(f"  RMSE: {quality['rmse']:.4f} pixels")
        print(f"  Mean error: {quality['mean_error']:.4f} pixels")
        print(f"  Max error: {quality['max_error']:.4f} pixels")

        # Show displacement precision
        fractional_count = sum(
            1 for dx, dy in displacements if dx != int(dx) or dy != int(dy)
        )
        print(f"  Fractional displacements: {fractional_count}/{len(displacements)}")


def robustness_testing():
    """Test robustness to different noise levels and conditions."""
    print("\n\nRobustness Testing")
    print("=" * 50)

    base_params = {"n_slices": 5, "height": 120, "width": 120, "translation_range": 3.0}

    bbox = (40, 40, 40, 40)
    config = AlignmentConfig(method=5, subpixel=True)

    # Test different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

    print("Noise Level | RMSE  | Success Rate")
    print("-" * 35)

    for noise_level in noise_levels:
        rmse_values = []
        success_count = 0
        trials = 5

        for trial in range(trials):
            try:
                # Create test data with different random seed
                np.random.seed(42 + trial)
                image_stack, true_displacements = create_test_image_stack(
                    noise_level=noise_level, **base_params
                )

                aligned_stack, displacements = register_stack(
                    image_stack, bbox, reference_slice=0, config=config
                )

                quality = calculate_alignment_quality(displacements, true_displacements)
                rmse_values.append(quality["rmse"])

                if quality["rmse"] < 2.0:  # Reasonable threshold
                    success_count += 1

            except Exception:
                pass

        avg_rmse = np.mean(rmse_values) if rmse_values else float("inf")
        success_rate = success_count / trials * 100

        print(f"{noise_level:<11.2f} | {avg_rmse:<5.2f} | {success_rate:<11.0f}%")


def performance_profiling():
    """Profile performance with different stack sizes."""
    print("\n\nPerformance Profiling")
    print("=" * 50)

    sizes = [
        (64, 64, 5),  # Small
        (128, 128, 8),  # Medium
        (256, 256, 10),  # Large
    ]

    print("Stack Size    | Slices | Template | Time (s) | Time/Slice")
    print("-" * 60)

    for width, height, n_slices in sizes:
        # Create test data
        image_stack, _ = create_test_image_stack(
            n_slices=n_slices,
            height=height,
            width=width,
            translation_range=3.0,
            noise_level=0.1,
        )

        # Use proportional template size
        template_size = min(width, height) // 4
        x = (width - template_size) // 2
        y = (height - template_size) // 2
        bbox = (x, y, template_size, template_size)

        config = AlignmentConfig(method=5, subpixel=True)

        start_time = time.time()
        aligned_stack, displacements = register_stack(
            image_stack, bbox, reference_slice=0, config=config
        )
        total_time = time.time() - start_time
        time_per_slice = total_time / n_slices

        print(
            f"{width}x{height:<7} | {n_slices:<6} | {template_size}x{template_size:<6} | "
            f"{total_time:<8.3f} | {time_per_slice:<10.3f}"
        )


def main():
    """Run all advanced alignment demonstrations."""
    print("TemplateMatchingPy - Advanced Alignment Examples")
    print("=" * 60)

    # Run all demonstrations
    compare_alignment_methods()
    demonstrate_search_area_effects()
    subpixel_precision_analysis()
    robustness_testing()
    performance_profiling()

    print("\n" + "=" * 60)
    print("Advanced alignment examples completed!")
    print("\nKey takeaways:")
    print("- TM_CCOEFF_NORMED (method 5) generally performs best")
    print("- Sub-pixel precision improves accuracy significantly")
    print("- Search area restriction can speed up processing")
    print("- Method choice affects both accuracy and speed")
    print("- Higher noise levels reduce alignment accuracy")


if __name__ == "__main__":
    main()
