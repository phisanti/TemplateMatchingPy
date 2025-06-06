# Advanced Examples: Tutorial

This section provides in-depth tutorials for advanced TemplateMatchingPy workflows. These examples go beyond the basics to help you understand performance, robustness, and method selection for challenging scenarios.

---

## Example 1: Comparing Template Matching Methods

**When to use:**
- You want to benchmark different OpenCV template matching methods.
- You need to select the best method for your data.

**What you'll learn:**
- How to compare accuracy and speed of different methods
- How to interpret quality metrics (RMSE, mean error)

### Step-by-Step
1. **Create a synthetic stack:**
   ```python
   image_stack, true_displacements = create_test_image_stack(n_slices=6, height=200, width=200, translation_range=4.0, noise_level=0.15)
   bbox = (75, 75, 50, 50)
   ```
2. **Test multiple methods:**
   ```python
   methods = {1: "TM_SQDIFF_NORMED", 3: "TM_CCORR_NORMED", 4: "TM_CCOEFF", 5: "TM_CCOEFF_NORMED"}
   for method_id, method_name in methods.items():
       config = AlignmentConfig(method=method_id, subpixel=True)
       aligned_stack, displacements = register_stack(image_stack, bbox, reference_slice=0, config=config)
       quality = calculate_alignment_quality(displacements, true_displacements)
       print(f"{method_name}: RMSE={quality['rmse']:.3f}")
   ```
3. **Interpret results:**
   Lower RMSE and mean error indicate better alignment. TM_CCOEFF_NORMED (method 5) is often best.

---

## Example 2: Search Area Effects

**When to use:**
- You want to optimize speed or handle large translations.

**What you'll learn:**
- How restricting the search area affects speed and accuracy

### Step-by-Step
1. **Test different search areas:**
   ```python
   for search_area in [0, 5, 10, 20, 30]:
       config = AlignmentConfig(method=5, subpixel=True, search_area=search_area)
       aligned_stack, displacements = register_stack(image_stack, bbox, reference_slice=0, config=config)
       quality = calculate_alignment_quality(displacements, true_displacements)
       print(f"Search area {search_area}: RMSE={quality['rmse']:.2f}")
   ```
2. **Interpret results:**
   Smaller search areas are faster but may miss large shifts.

---

## Example 3: Sub-pixel Precision Analysis

**When to use:**
- You need high-accuracy alignment for quantitative analysis.

**What you'll learn:**
- The impact of sub-pixel refinement on alignment accuracy

### Step-by-Step
1. **Compare integer vs sub-pixel:**
   ```python
   configs = {"Integer": AlignmentConfig(method=5, subpixel=False), "Sub-pixel": AlignmentConfig(method=5, subpixel=True)}
   for name, config in configs.items():
       aligned_stack, displacements = register_stack(image_stack, bbox, reference_slice=0, config=config)
       quality = calculate_alignment_quality(displacements, true_displacements)
       print(f"{name}: RMSE={quality['rmse']:.4f}")
   ```
2. **Interpret results:**
   Sub-pixel mode should yield lower RMSE and more precise displacements.

---

## Example 4: Robustness Testing

**When to use:**
- You want to know how noise affects alignment.

**What you'll learn:**
- How alignment accuracy degrades with increasing noise

### Step-by-Step
1. **Vary noise levels:**
   ```python
   for noise_level in [0.0, 0.05, 0.1, 0.2, 0.3]:
       image_stack, true_displacements = create_test_image_stack(noise_level=noise_level, n_slices=5, height=120, width=120, translation_range=3.0)
       aligned_stack, displacements = register_stack(image_stack, bbox, reference_slice=0, config=config)
       quality = calculate_alignment_quality(displacements, true_displacements)
       print(f"Noise {noise_level}: RMSE={quality['rmse']:.2f}")
   ```
2. **Interpret results:**
   Higher noise increases RMSE and reduces success rate.

---

## Example 5: Performance Profiling

**When to use:**
- You need to estimate runtime for large datasets.

**What you'll learn:**
- How stack size and template size affect speed

### Step-by-Step
1. **Profile different sizes:**
   ```python
   for width, height, n_slices in [(64,64,5), (128,128,8), (256,256,10)]:
       image_stack, _ = create_test_image_stack(n_slices=n_slices, height=height, width=width, translation_range=3.0, noise_level=0.1)
       template_size = min(width, height) // 4
       bbox = ((width-template_size)//2, (height-template_size)//2, template_size, template_size)
       config = AlignmentConfig(method=5, subpixel=True)
       start = time.time()
       aligned_stack, displacements = register_stack(image_stack, bbox, reference_slice=0, config=config)
       print(f"{width}x{height}, {n_slices} slices: {time.time()-start:.2f}s")
   ```
2. **Interpret results:**
   Larger stacks and templates take longer; optimize for your use case.

---

## Tips & Takeaways
- TM_CCOEFF_NORMED (method 5) is generally best for most data.
- Sub-pixel precision is recommended for quantitative work.
- Restrict search area for speed, but ensure it covers expected shifts.
- Test with your own data and noise levels to validate robustness.

See also: [Basic Examples](basic-examples.md) and [Performance Tuning](../guides/performance-tuning.md).
