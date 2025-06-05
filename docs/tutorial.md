# TemplateMatchingPy Tutorial

This tutorial provides a comprehensive guide to using TemplateMatchingPy for image stack alignment and template matching in microscopy workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Concepts](#basic-concepts)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

```bash
pip install templatematchingpy
```

### Basic Import

```python
import numpy as np
from templatematchingpy import register_stack, AlignmentConfig
```

## Basic Concepts

### What is Template Matching?

Template matching is a technique used to find regions in an image that match a template image. In the context of image stack alignment, we:

1. Select a template region from a reference slice
2. Find the best matching location in other slices
3. Calculate the displacement needed to align each slice
4. Apply the translation to create an aligned stack

### Key Components

- **Image Stack**: 3D array (slices, height, width) containing the image sequence
- **Template**: Small region selected from the reference slice
- **Bounding Box**: Coordinates defining the template region (x, y, width, height)
- **Reference Slice**: The slice that other slices are aligned to (usually slice 0)
- **Displacements**: Translation vectors (dx, dy) for each slice

## Step-by-Step Tutorial

### Step 1: Prepare Your Data

First, load your image stack. The stack should be a 3D numpy array:

```python
import numpy as np
from templatematchingpy import load_image_stack

# Load from file
image_stack = load_image_stack("your_stack.tiff")

# Or create synthetic data for testing
from templatematchingpy import create_test_image_stack
image_stack, true_displacements = create_test_image_stack(
    n_slices=10,
    height=256,
    width=256,
    translation_range=5.0
)

print(f"Stack shape: {image_stack.shape}")
```

### Step 2: Choose Template Region

Select a region that:
- Has good contrast and distinctive features
- Is present in all slices
- Is not too close to image borders

```python
# Define bounding box: (x, y, width, height)
# Example: 64x64 pixel region starting at (100, 100)
bbox = (100, 100, 64, 64)

# Visualize the template region (optional)
import matplotlib.pyplot as plt

reference_slice = 0
template = image_stack[reference_slice, 
                     bbox[1]:bbox[1]+bbox[3], 
                     bbox[0]:bbox[0]+bbox[2]]

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image_stack[reference_slice], cmap='gray')
plt.title('Reference Slice')
plt.subplot(132)
plt.imshow(template, cmap='gray')
plt.title('Template Region')
plt.show()
```

### Step 3: Configure Alignment

```python
from templatematchingpy import AlignmentConfig

# Basic configuration (recommended for most cases)
config = AlignmentConfig(
    method=5,        # TM_CCOEFF_NORMED (best for most cases)
    subpixel=True,   # Enable sub-pixel precision
    search_area=0    # Search entire image
)

# Alternative configurations:
# For faster processing with limited search area:
config_fast = AlignmentConfig(method=5, search_area=50)

# For integer-only alignment:
config_integer = AlignmentConfig(method=5, subpixel=False)
```

### Step 4: Perform Alignment

```python
from templatematchingpy import register_stack

# Align the stack
aligned_stack, displacements = register_stack(
    image_stack=image_stack,
    bbox=bbox,
    reference_slice=0,  # Use first slice as reference
    config=config
)

print(f"Alignment completed!")
print(f"Original stack shape: {image_stack.shape}")
print(f"Aligned stack shape: {aligned_stack.shape}")
print(f"Number of displacements: {len(displacements)}")
```

### Step 5: Analyze Results

```python
# Display displacements
print("Calculated displacements:")
for i, (dx, dy) in enumerate(displacements):
    print(f"Slice {i:2d}: dx={dx:6.2f}, dy={dy:6.2f}")

# Calculate quality metrics
from templatematchingpy import calculate_alignment_quality

quality = calculate_alignment_quality(displacements)
print(f"\nAlignment Quality:")
print(f"Mean displacement: {quality['mean_displacement']:.3f} pixels")
print(f"Max displacement: {quality['max_displacement']:.3f} pixels")
print(f"Displacement range (dx): {quality['dx_range']:.3f} pixels")
print(f"Displacement range (dy): {quality['dy_range']:.3f} pixels")
```

### Step 6: Visualize Results

```python
import matplotlib.pyplot as plt

# Compare before and after alignment
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

slices_to_show = [0, len(image_stack)//2, -1]

for i, slice_idx in enumerate(slices_to_show):
    # Original
    axes[0, i].imshow(image_stack[slice_idx], cmap='gray')
    axes[0, i].set_title(f'Original Slice {slice_idx}')
    axes[0, i].axis('off')
    
    # Aligned
    axes[1, i].imshow(aligned_stack[slice_idx], cmap='gray')
    axes[1, i].set_title(f'Aligned Slice {slice_idx}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# Plot displacement vectors
plt.figure(figsize=(10, 6))
dx_values = [d[0] for d in displacements]
dy_values = [d[1] for d in displacements]

plt.subplot(121)
plt.plot(dx_values, 'b.-', label='dx')
plt.plot(dy_values, 'r.-', label='dy')
plt.xlabel('Slice Index')
plt.ylabel('Displacement (pixels)')
plt.title('Displacement Vectors')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(dx_values, dy_values, 'go-')
plt.xlabel('dx (pixels)')
plt.ylabel('dy (pixels)')
plt.title('Displacement Trajectory')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()
```

### Step 7: Save Results

```python
from templatematchingpy import save_image_stack

# Save aligned stack
save_image_stack(aligned_stack, "aligned_stack.tiff")

# Save displacement data
import csv
with open("displacements.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["slice", "dx", "dy"])
    for i, (dx, dy) in enumerate(displacements):
        writer.writerow([i, dx, dy])

print("Results saved!")
```

## Advanced Features

### Different Template Matching Methods

```python
# Compare different methods
methods = {
    1: "TM_SQDIFF_NORMED",
    3: "TM_CCORR_NORMED", 
    5: "TM_CCOEFF_NORMED"
}

results = {}
for method_id, method_name in methods.items():
    config = AlignmentConfig(method=method_id, subpixel=True)
    _, displacements = register_stack(image_stack, bbox, config=config)
    
    quality = calculate_alignment_quality(displacements)
    results[method_name] = quality
    
    print(f"{method_name}: RMSE = {quality['mean_displacement']:.3f}")
```

### Search Area Optimization

```python
# Test different search areas for speed vs accuracy trade-off
search_areas = [0, 20, 50, 100]  # 0 = full image

for search_area in search_areas:
    config = AlignmentConfig(method=5, search_area=search_area)
    
    import time
    start_time = time.time()
    _, displacements = register_stack(image_stack, bbox, config=config)
    processing_time = time.time() - start_time
    
    quality = calculate_alignment_quality(displacements)
    
    area_desc = "Full image" if search_area == 0 else f"{search_area} pixels"
    print(f"Search area {area_desc}: "
          f"RMSE={quality['mean_displacement']:.3f}, "
          f"Time={processing_time:.2f}s")
```

### Sub-pixel vs Integer Precision

```python
# Compare sub-pixel vs integer precision
configs = {
    'Integer': AlignmentConfig(method=5, subpixel=False),
    'Sub-pixel': AlignmentConfig(method=5, subpixel=True)
}

for name, config in configs.items():
    _, displacements = register_stack(image_stack, bbox, config=config)
    quality = calculate_alignment_quality(displacements)
    
    print(f"{name} precision: RMSE = {quality['mean_displacement']:.4f}")
```

### Batch Processing

```python
# Process multiple stacks
from pathlib import Path

input_dir = Path("input_stacks")
output_dir = Path("output_stacks")
output_dir.mkdir(exist_ok=True)

config = AlignmentConfig(method=5, subpixel=True)

for input_file in input_dir.glob("*.tiff"):
    print(f"Processing {input_file.name}...")
    
    # Load stack
    stack = load_image_stack(str(input_file))
    
    # Align
    aligned_stack, displacements = register_stack(
        stack, bbox, config=config
    )
    
    # Save
    output_file = output_dir / f"aligned_{input_file.name}"
    save_image_stack(aligned_stack, str(output_file))
    
    print(f"  Saved to {output_file}")
```

## Best Practices

### 1. Template Selection

- **Size**: 32-128 pixels square typically works well
- **Features**: Choose regions with:
  - High contrast
  - Distinctive patterns
  - Good signal-to-noise ratio
- **Avoid**: Edges, homogeneous regions, areas with artifacts

### 2. Method Selection

- **TM_CCOEFF_NORMED (method=5)**: Best for most applications
- **TM_SQDIFF_NORMED (method=1)**: Good for high-contrast templates
- **TM_CCORR_NORMED (method=3)**: Robust to illumination changes

### 3. Performance Optimization

- Use `search_area` parameter for known small displacements
- Disable `subpixel` for faster integer-only alignment
- Choose smaller template sizes for speed

### 4. Quality Assessment

Always check alignment quality:

```python
quality = calculate_alignment_quality(displacements)

# Warning thresholds
if quality['mean_displacement'] > 10:
    print("Warning: Large displacements detected")
if quality['max_displacement'] > 50:
    print("Warning: Very large displacement detected")
```

### 5. Error Handling

```python
try:
    aligned_stack, displacements = register_stack(
        image_stack, bbox, reference_slice=0, config=config
    )
except ValueError as e:
    print(f"Invalid input parameters: {e}")
except IndexError as e:
    print(f"Index error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Troubleshooting

### Common Issues

#### 1. Poor Alignment Quality

**Symptoms**: Large RMSE, inconsistent displacements

**Solutions**:
- Choose a better template region with more distinctive features
- Increase template size
- Try different matching methods
- Check for image artifacts or blur

#### 2. Template Not Found

**Symptoms**: Very large displacements, alignment fails

**Solutions**:
- Verify template region exists in all slices
- Increase search area
- Check image preprocessing (contrast, noise)

#### 3. Slow Performance

**Symptoms**: Long processing times

**Solutions**:
- Reduce template size
- Use search area parameter
- Disable sub-pixel precision if not needed
- Consider different matching method

#### 4. Memory Issues

**Symptoms**: Out of memory errors with large stacks

**Solutions**:
- Process stacks in batches
- Reduce image resolution if possible
- Use smaller template sizes

### Validation Examples

```python
# Validate input data
from templatematchingpy.utils import validate_image_stack, validate_bbox

try:
    validate_image_stack(image_stack)
    validate_bbox(bbox, image_stack.shape[1:])
    print("Input validation passed")
except ValueError as e:
    print(f"Validation error: {e}")

# Test with known synthetic data
test_stack, true_displacements = create_test_image_stack(
    n_slices=5, translation_range=3.0
)

bbox_test = (50, 50, 40, 40)
_, calc_displacements = register_stack(test_stack, bbox_test)

# Compare results
quality = calculate_alignment_quality(calc_displacements, true_displacements)
print(f"Synthetic data RMSE: {quality['rmse']:.3f} pixels")
```

This tutorial covers the essential aspects of using TemplateMatchingPy. For more detailed information, see the [API Reference](api_reference.md) and example scripts in the `examples/` directory.
