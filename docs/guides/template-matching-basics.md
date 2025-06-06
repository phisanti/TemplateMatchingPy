# Template Matching Fundamentals

This guide provides a comprehensive introduction to template matching and its application to image stack alignment. It covers the core theory, the overall algorithm, and the big-picture ideas behind why and how template matching is used for registration. For configuration details and parameter tuning, see the [Configuration Guide](configuration-guide.md).

---

## What is Template Matching?
Template matching is a classical computer vision technique for finding the location of a small image (the template) within a larger image (the source). It works by sliding the template over the source and computing a similarity score at each position. The position with the highest (or lowest) score is considered the best match.

### Why Use Template Matching for Stack Alignment?
In image stack alignment, the goal is to register a sequence of images (e.g., slices in a microscopy stack) so that features are spatially consistent across slices. Template matching is ideal for translation-only alignment, offering:

- Simplicity and speed
- No need for feature extraction or training
- Sub-pixel accuracy with refinement

**Applications:**

- Microscopy (z-stacks, time-lapse)
- Medical imaging (histology and histopathology)
- Materials science, remote sensing

---

## Core Concepts and Terminology
- **Template:** The region of interest (ROI) to be matched, typically a patch from a reference image.
- **Source:** The image (or slice) in which the template is searched.
- **Displacement:** The (dx, dy) translation needed to align the template with the best-matching region in the source.
- **Correlation Map:** The result of the matching process, showing similarity scores for each possible template position.
- **Reference Slice:** The image in the stack used as the alignment anchor.

### Coordinate Systems
- Displacements are reported as (dx, dy) in pixel units.
- All coordinates are relative to the top-left corner of the image.

---

## The Template Matching Algorithm: Step by Step

The template matching process follows a systematic approach to find the best alignment between images. Understanding this workflow is essential for effective use of the algorithm and troubleshooting alignment issues.

**High-Level Algorithm:**
```
ALGORITHM: Template Matching for Image Alignment
INPUT: reference_image, target_image, template_bounds, search_bounds
OUTPUT: displacement (dx, dy)

1. template = EXTRACT_REGION(reference_image, template_bounds)
2. search_region = EXTRACT_REGION(target_image, search_bounds)
3. correlation_map = COMPUTE_CORRELATION(template, search_region)
4. peak_position = FIND_PEAK(correlation_map)
5. IF sub_pixel_enabled THEN
    peak_position = GAUSSIAN_FIT(correlation_map, peak_position)
6. displacement = CALCULATE_OFFSET(peak_position, template_bounds, search_bounds)
7. RETURN displacement
```

This pseudocode translates into the following detailed steps:

1. **Extract Template:** Select a region (bounding box) from the reference image.
2. **Define Search Region:** Optionally restrict the area in the target image to speed up matching.
3. **Compute Correlation:** Slide the template over the search region, computing a similarity score at each position.
4. **Find Peak:** Locate the position with the best score (max or min, depending on method).
5. **Sub-pixel Refinement:** Optionally fit a Gaussian to the peak for higher accuracy.
6. **Calculate Displacement:** Compute the translation needed to align the template with the best match.

---

## Sub-pixel Accuracy and Gaussian Peak Fitting
- **Why Sub-pixel?**
  - Integer-pixel alignment may not be sufficient for quantitative analysis.
  - Sub-pixel refinement fits a 2D Gaussian to the correlation peak, yielding (dx, dy) with decimal precision.
- **When to Use:**
  - Always enable for high-precision tasks (e.g., quantitative microscopy).
  - May be unnecessary for rough alignment or low-resolution data.

---

## Performance and Limitations
- **Speed:** Template matching is fast for moderate image sizes and can be accelerated with GPU (see [OpenCV CUDA Guide](https://docs.opencv.org/3.4/d6/d15/tutorial_building_tegra_cuda.html)).
- **Limitations:**
  - Sensitive to large content changes, occlusions, or non-rigid deformations
  - Requires careful template and search region selection for best results
  - Computational cost increases with image/search area size

---

## Common Pitfalls and Troubleshooting
- **Poor Alignment:**
  - Template is too small or lacks unique features.
  - Search area is too small, missing the true match.
  - Large intensity or contrast changes between slices.
- **Numerical Issues:**
  - Correlation map is flat (no clear peak): try a different template or method.
  - Sub-pixel fitting fails: check for boundary effects or try disabling sub-pixel mode.
- **Boundary Effects:**
  - Template or search region extends beyond image borders.
  - Always check that your bounding box is valid for all slices.

---

## Further Reading
- [Configuration Guide](configuration-guide.md): All configuration options explained
- [Performance Tuning](performance-tuning.md): Advanced optimization strategies

For more on the underlying algorithms, see the [OpenCV documentation](https://docs.opencv.org/4.x/d4/dc6/tutorial_template_matching.html).
