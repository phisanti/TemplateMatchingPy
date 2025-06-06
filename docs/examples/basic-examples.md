# Basic Examples: Tutorial

This section provides step-by-step tutorials for the most common TemplateMatchingPy workflows. Each example explains not just *how* to use the library, but also *when* and *why* you might choose a particular approach.

---

## Example 1: Aligning a Synthetic Image Stack

**When to use:**
- You want to test TemplateMatchingPy without real data.
- You need to validate installation or benchmark accuracy.

**What you'll learn:**
- How to create a synthetic stack
- How to define a template region
- How to configure and run alignment
- How to interpret results and errors

### Step-by-Step

1. **Create a synthetic stack:**
   ```python
   from templatematchingpy import create_test_image_stack
   image_stack, true_displacements = create_test_image_stack(
       n_slices=8, height=256, width=256, translation_range=5.0, noise_level=0.1
   )
   ```
   This generates a stack with known translations and noise, so you can compare true and calculated displacements.

2. **Define the template region:**
   ```python
   bbox = (100, 100, 56, 56)  # (x, y, width, height)
   ```
   The template should cover a region with distinctive features. Avoid uniform or low-contrast areas.

3. **Configure alignment:**
   ```python
   from templatematchingpy import AlignmentConfig
   config = AlignmentConfig(method=5, subpixel=True, search_area=0)
   ```
   - `method=5` uses normalized cross-correlation (recommended for most cases).
   - `subpixel=True` enables sub-pixel accuracy.
   - `search_area=0` searches the whole image (slower, but robust).

4. **Run the alignment:**
   ```python
   from templatematchingpy import register_stack
   aligned_stack, displacements = register_stack(
       image_stack=image_stack, bbox=bbox, reference_slice=0, config=config
   )
   ```
   The function returns the aligned stack and a list of (dx, dy) displacements for each slice.

5. **Evaluate results:**
   Compare calculated displacements to the known true values:
   ```python
   for i, (true, calc) in enumerate(zip(true_displacements, displacements)):
       print(f"Slice {i}: True {true}, Calculated {calc}")
   ```
   For synthetic data, errors should be small (typically <1 pixel with subpixel enabled).

6. **(Optional) Save results:**
   ```python
   from templatematchingpy import save_image_stack
   save_image_stack(aligned_stack, "aligned_stack.tiff")
   ```

---

## Example 2: Aligning a Real Image Stack (TIFF)

**When to use:**
- You have a multi-page TIFF stack from microscopy or other imaging.
- You want to align real experimental data.

**What you'll learn:**
- How to load a TIFF stack
- How to select a template region
- How to save aligned results

### Step-by-Step

1. **Load the TIFF stack:**
   ```python
   import cv2
   ret, images = cv2.imreadmulti("./examples/data/example_image_stack.tiff", flags=cv2.IMREAD_GRAYSCALE)
   if not ret:
       raise ValueError("Could not load TIFF stack")
   image_stack = np.array(images, dtype=np.float32)
   if image_stack.max() > 1.0:
       image_stack = image_stack / 255.0
   ```

2. **Define the template region:**
   ```python
   height, width = image_stack.shape[1], image_stack.shape[2]
   bbox = ((width-100)//2, (height-100)//2, 100, 100)
   ```

3. **Configure and run alignment:**
   ```python
   config = AlignmentConfig(method=5, subpixel=True)
   aligned_stack, displacements = register_stack(image_stack, bbox=bbox, reference_slice=0, config=config)
   ```

4. **Save the aligned stack:**
   ```python
   aligned_frames = [frame.astype(np.float32) for frame in aligned_stack]
   cv2.imwritemulti("aligned_stack.tiff", aligned_frames)
   ```

5. **Inspect results:**
   Print or plot the displacements to check for outliers or drift.

---

## Tips & Troubleshooting
- If alignment fails, try increasing the template size or adjusting the search area.
- For best results, the template should contain unique, high-contrast features.
- If your stack is very large, consider cropping or downsampling for speed.

See also: [Quick Start](../getting-started/quick-start.md) and [Advanced Examples](advanced-examples.md) for more scenarios.
