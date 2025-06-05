"""Basic usage example for TemplateMatchingPy.

This example demonstrates the most common use case: aligning an image stack
using template matching with a specified region of interest.
"""

import numpy as np
import cv2
from templatematchingpy import register_stack, AlignmentConfig, create_test_image_stack


def main():
    """Demonstrate basic stack alignment workflow."""
    print("TemplateMatchingPy - Basic Usage Example")
    print("=" * 40)

    # Create a synthetic image stack for demonstration
    print("Creating synthetic image stack...")
    image_stack, true_displacements = create_test_image_stack(
        n_slices=8, height=256, width=256, translation_range=5.0, noise_level=0.1
    )
    # Validate the image stack
    print(f"Created stack with shape: {image_stack.shape}")
    print(f"True displacements: {true_displacements[:3]}...")  # Show first 3

    # Define template region (bounding box)
    # Format: (x, y, width, height)
    bbox = (100, 100, 56, 56)
    print(f"Template bounding box: {bbox}")

    # Create alignment configuration
    config = AlignmentConfig(
        method=5,  # TM_CCOEFF_NORMED (recommended)
        subpixel=True,  # Enable sub-pixel precision
        search_area=0,  # Search entire image
    )
    print(
        f"Alignment configuration: method={config.method}, subpixel={config.subpixel}"
    )

    # Perform stack registration
    print("\nPerforming stack alignment...")
    aligned_stack, displacements = register_stack(
        image_stack=image_stack,
        bbox=bbox,
        reference_slice=0,  # Use first slice as reference
        config=config,
    )

    print("Alignment completed!")
    print(f"Aligned stack shape: {aligned_stack.shape}")
    print(f"Number of displacement vectors: {len(displacements)}")

    # Display results
    print("\nCalculated displacements:")
    for i, (dx, dy) in enumerate(displacements):
        print(f"  Slice {i}: dx={dx:6.2f}, dy={dy:6.2f}")

    # Compare with true displacements (for synthetic data)
    print("\nComparison with true displacements:")
    print("Slice |   True (dx, dy)   |  Calculated (dx, dy) |  Error")
    print("-" * 60)

    total_error = 0.0
    for i, ((true_dx, true_dy), (calc_dx, calc_dy)) in enumerate(
        zip(true_displacements, displacements)
    ):
        if i == 0:  # Reference slice
            error = 0.0
        else:
            # Note: calculated displacement is negative of applied translation
            error_dx = calc_dx - (-true_dx)
            error_dy = calc_dy - (-true_dy)
            error = np.sqrt(error_dx**2 + error_dy**2)
            total_error += error

        print(
            f"  {i:2d}  | ({true_dx:6.2f}, {true_dy:6.2f}) | "
            f"({calc_dx:6.2f}, {calc_dy:6.2f}) | {error:6.3f}"
        )

    avg_error = total_error / (len(displacements) - 1)  # Exclude reference slice
    print(f"\nAverage alignment error: {avg_error:.3f} pixels")

    # Calculate alignment quality metrics
    from templatematchingpy import calculate_alignment_quality

    quality = calculate_alignment_quality(displacements, true_displacements)
    print(f"\nAlignment quality metrics:")
    print(f"  Mean displacement: {quality['mean_displacement']:.3f} pixels")
    print(f"  RMSE: {quality['rmse']:.3f} pixels")
    print(f"  Maximum error: {quality['max_error']:.3f} pixels")

    # Demonstrate saving results (optional)
    save_results = False  # Set to True to save files
    if save_results:
        from templatematchingpy import save_image_stack

        print("\nSaving results...")
        save_image_stack(image_stack, "original_stack.tiff")
        save_image_stack(aligned_stack, "aligned_stack.tiff")
        print("Saved original_stack.tiff and aligned_stack.tiff")

    print("\nBasic usage example completed successfully!")


if __name__ == "__main__":
    main()
