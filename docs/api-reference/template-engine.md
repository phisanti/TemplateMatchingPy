# TemplateMatchingEngine Class Reference

## Class Signature
```python
class TemplateMatchingEngine:
    def __init__(self):
        ...
```

---

## Methods

### __init__()
Initializes the template matching engine.

#### Arguments
- None

#### Returns
- None

#### Details
- Initializes the set of supported OpenCV methods (0–5).

#### Example
```python
engine = TemplateMatchingEngine()
```

---

### match_template(source, template, method)
Perform template matching using the specified OpenCV method.

#### Arguments
- **source** (`np.ndarray`): Source image array (2D, grayscale)
- **template** (`np.ndarray`): Template image array (2D, grayscale)
- **method** (`int`): OpenCV matching method (0–5)

#### Returns
- **correlation_map** (`np.ndarray`): 2D float32 array of similarity scores

#### Raises
- `ValueError`: If method is not supported or template is larger than source

#### Details
- Converts images to float32 and normalizes if needed
- Uses `cv2.matchTemplate` with the selected method
- See [OpenCV docs](https://docs.opencv.org/4.x/d4/dc6/tutorial_template_matching.html) for method details

#### Example
```python
result = engine.match_template(source, template, method=5)
```

---

### find_peak(correlation_map, method)
Find the peak location in a correlation map.

#### Arguments
- **correlation_map** (`np.ndarray`): Output from `match_template`
- **method** (`int`): OpenCV matching method (affects min/max selection)

#### Returns
- **(x, y)** (`Tuple[int, int]`): Integer coordinates of the peak

#### Details
- For methods 0/1 (SQDIFF), finds the minimum; for others, finds the maximum
- Uses `cv2.minMaxLoc` internally

#### Example
```python
peak = engine.find_peak(result, method=5)
# peak = (x, y)
```

---

### gaussian_peak_fit(correlation_map, x, y)
Refine the peak location to sub-pixel accuracy using Gaussian fitting.

#### Arguments
- **correlation_map** (`np.ndarray`): Correlation map from `match_template`
- **x** (`int`): Integer x-coordinate of the peak
- **y** (`int`): Integer y-coordinate of the peak

#### Returns
- **(x_refined, y_refined)** (`Tuple[float, float]`): Sub-pixel peak coordinates

#### Details
- Uses logarithmic parabolic (Gaussian) fitting on the peak and its neighbors
- If the peak is at the boundary or fitting fails, returns the integer coordinates
- The sub-pixel offset is computed as:
  $$
  \Delta x = \frac{\log(L) - \log(R)}{2\log(L) - 4\log(C) + 2\log(R)}
  $$
  where $L$, $C$, $R$ are the left, center, and right values (with $\epsilon$ for stability)
- Offsets are clipped to $[-1, 1]$ for robustness

#### Example
```python
x, y = engine.find_peak(result, method=5)
x_sub, y_sub = engine.gaussian_peak_fit(result, x, y)
```

---

See also: [StackAligner](stack-aligner.md), [Template Matching Fundamentals](../guides/template-matching-basics.md)
