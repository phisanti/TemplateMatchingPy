# Configuration Reference

## AlignmentConfig Class Reference

### Class Signature
```python
@dataclass
class AlignmentConfig:
    method: int = 5
    search_area: int = 0
    subpixel: bool = True
    interpolation: int = cv2.INTER_LINEAR
```

---

### Fields
- `method` (`int`): OpenCV template matching method (0–5)
    - 0: `TM_SQDIFF`
    - 1: `TM_SQDIFF_NORMED`
    - 2: `TM_CCORR`
    - 3: `TM_CCORR_NORMED`
    - 4: `TM_CCOEFF`
    - 5: `TM_CCOEFF_NORMED` (**default, recommended**)
- `search_area` (`int`): Number of pixels around the ROI to search for matches. `0` means search the entire image (default).
- `subpixel` (`bool`): Enable sub-pixel registration using Gaussian fitting (default: `True`).
- `interpolation` (`int`): OpenCV interpolation method for image warping (default: `cv2.INTER_LINEAR`).

---

### Methods
#### `__post_init__()`
Validates configuration parameters after initialization.
- **Raises:**
    - `ValueError`: If `method` is not in 0–5 or `search_area` is negative.

---

### Example
```python
from templatematchingpy.core.config import AlignmentConfig
config = AlignmentConfig(method=5, search_area=10, subpixel=True)
```

---

## Constants

### `MATCHING_METHODS`
A dictionary mapping method indices to OpenCV constants:
```python
MATCHING_METHODS = {
    0: cv2.TM_SQDIFF,
    1: cv2.TM_SQDIFF_NORMED,
    2: cv2.TM_CCORR,
    3: cv2.TM_CCORR_NORMED,
    4: cv2.TM_CCOEFF,
    5: cv2.TM_CCOEFF_NORMED,
}
```
- Use the `method` field in `AlignmentConfig` to select the matching method.

### `MIN_VALUE_METHODS`
A set of method indices for which the minimum value in the correlation map indicates the best match (i.e., methods 0 and 1):
```python
MIN_VALUE_METHODS = {0, 1}
```
- For these methods, the best match is found by minimizing the score; for others, by maximizing.

### `EPS`
A small constant to avoid division by zero or log of zero in calculations:
```python
EPS = 1e-10
```
- Used in sub-pixel Gaussian fitting and normalization steps.

---

See also: [Configuration Guide](../guides/configuration-guide.md)
