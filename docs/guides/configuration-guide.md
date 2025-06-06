# Configuration Guide

This guide provides a detailed explanation of all configuration options in TemplateMatchingPy, focusing on how to control and tune the alignment process for your specific data and use case. It is intended as a practical reference for users who want to understand and optimize the behavior of the library.

---

## Overview: When to Change Settings
- **Defaults:** The default configuration is robust for most use cases. However, visual inspection of results is always recommended.
- **When to adjust:** Change settings if you observe poor alignment, have special requirements (e.g., speed, precision), or are working with unusual data (e.g., very large images, low contrast, or drifting objects).

---

## AlignmentConfig Parameters and Defaults

### `method` (OpenCV Matching Method)
- **Description:** Selects the template matching algorithm.
- **Options:**
    - `0`: TM_SQDIFF (Sum of squared differences)
    - `1`: TM_SQDIFF_NORMED (Normalized SSD)
    - `2`: TM_CCORR (Cross-correlation)
    - `3`: TM_CCORR_NORMED (Normalized cross-correlation)
    - `4`: TM_CCOEFF (Correlation coefficient)
    - `5`: TM_CCOEFF_NORMED (**default, recommended**)
- **Best Practice:** Use `TM_CCOEFF_NORMED` (5) for most applications. See [OpenCV docs](https://docs.opencv.org/4.x/d4/dc6/tutorial_template_matching.html) for details.

### `search_area`
- **Description:** Number of pixels around the template region to search for matches. `0` means search the entire image (most robust, slowest).
- **When to change:** Increase for large drifts, decrease for speed if drift is small.

### `subpixel`
- **Description:** Enable sub-pixel registration using Gaussian fitting for higher accuracy.
- **Default:** `True` (recommended for quantitative work)
- **When to change:** Disable only for speed or if sub-pixel accuracy is not needed.

### `interpolation`
- **Description:** OpenCV interpolation method for image warping (e.g., `cv2.INTER_LINEAR`).
- **Default:** `cv2.INTER_LINEAR`
- **When to change:** Use `cv2.INTER_NEAREST` for binary masks, or other methods for special cases.

---

## Reference Modes

### Static Reference
- **Use when:** The image is drifting in the XY plane but remains fairly constant over time.
- **How it works:** All slices are aligned to a single reference slice (usually the first or a manually chosen one).

### Dynamic Reference
- **Use when:** The image is drifting and also changing (e.g., growing tissue, moving colony).
- **How it works:** Each slice is aligned to the previous slice or a running average of previous slices. Set `reference_type='dynamic'` in the API.

---

## Bounding Box (Template Region) Selection
- **Centered bounding box:** Works well if the central region is present and stable across the stack.
- **ROI on object of interest:** Preferred if you want to track or align a specific feature.
- **Tips:**
    - The template should be large enough to contain unique, high-contrast features.
    - Avoid uniform or low-contrast regions.
    - Visualize the template and search region to ensure they are appropriate.

---

## Practical Examples

### Example: Custom Configuration
```python
from templatematchingpy.core.config import AlignmentConfig
config = AlignmentConfig(method=3, search_area=20, subpixel=True)
```

### Example: Static vs Dynamic Reference
```python
# Static reference (default)
aligned = aligner.register_stack(stack, bbox, reference_slice=0, reference_type='static')

# Dynamic reference (align to previous slice)
aligned = aligner.register_stack(stack, bbox, reference_slice=-1, reference_type='dynamic')
```

---

## Troubleshooting and Tips
- If alignment fails, try increasing the template size or search area.
- For best results, the template should contain unique, high-contrast features.
- If your stack is very large, consider cropping or downsampling for speed.
- Sub-pixel registration is computationally efficient and usually improves alignment.

---

See also: [Template Matching Fundamentals](template-matching-basics.md), [API Reference: Configuration](../api-reference/configuration.md)
