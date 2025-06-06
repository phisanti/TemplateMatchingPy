# Utility Functions Reference

## Image Validation and Processing Utilities

### validate_image_stack(image_stack)
Validate that the input is a proper 3D image stack.

#### Arguments
- **image_stack** (`np.ndarray`): Input array to validate

#### Raises
- `ValueError`: If the array is not a valid 3D image stack

#### Details
- Checks for correct type, dimensionality, and non-empty stack.

#### Example
```python
from templatematchingpy.utils.image_utils import validate_image_stack
validate_image_stack(stack)
```

---

### validate_bbox(bbox, image_shape)
Validate bounding box coordinates against image dimensions.

#### Arguments
- **bbox** (`Tuple[int, int, int, int]`): (x, y, width, height)
- **image_shape** (`Tuple[int, int]`): (height, width)

#### Raises
- `ValueError`: If bounding box is invalid

#### Details
- Ensures bbox is within image bounds and has positive size.

#### Example
```python
from templatematchingpy.utils.image_utils import validate_bbox
validate_bbox((10, 10, 50, 50), (100, 100))
```

---

### normalize_image(image, dtype=np.float32)
Normalize image to [0, 1] range and convert to specified dtype.

#### Arguments
- **image** (`np.ndarray`): Input image
- **dtype** (`np.dtype`, optional): Target data type (default: `np.float32`)

#### Returns
- **np.ndarray**: Normalized image

#### Details
- Converts to target dtype and rescales if needed.

#### Example
```python
from templatematchingpy.utils.image_utils import normalize_image
norm_img = normalize_image(img)
```

---

### extract_template(image, bbox)
Extract template region from image using bounding box.

#### Arguments
- **image** (`np.ndarray`): Source image
- **bbox** (`Tuple[int, int, int, int]`): (x, y, width, height)

#### Returns
- **np.ndarray**: Extracted template region

#### Raises
- `ValueError`: If bounding box is invalid

#### Example
```python
from templatematchingpy.utils.image_utils import extract_template
template = extract_template(img, (20, 20, 40, 40))
```

---

## Synthetic Data Generation

### create_test_image_stack(...)
Create a synthetic image stack for testing with known translations.

#### Arguments
- **n_slices** (`int`): Number of slices (default: 10)
- **height** (`int`): Image height (default: 256)
- **width** (`int`): Image width (default: 256)
- **noise_level** (`float`): Gaussian noise (0–1, default: 0.1)
- **translation_range** (`float`): Max translation in pixels (default: 5.0)
- **dtype** (`np.dtype`): Data type (default: `np.float32`)

#### Returns
- **Tuple[np.ndarray, List[Tuple[float, float]]]**: (image_stack, true_displacements)

#### Example
```python
from templatematchingpy.utils.image_utils import create_test_image_stack
stack, true_disp = create_test_image_stack(n_slices=5, noise_level=0.2)
```

---

## Quality Assessment

### calculate_alignment_quality(displacements, reference_displacements=None)
Calculate quality metrics for stack alignment.

#### Arguments
- **displacements** (`List[Tuple[float, float]]`): Calculated displacements
- **reference_displacements** (`List[Tuple[float, float]]`, optional): Known true displacements
- **invert_displacements**: If True, invert displacements (multiply by -1). This is useful because sometimes displacements are provided as the movement in the opposite direction than the true displacement.

#### Returns
- **dict**: Dictionary with quality metrics (mean, std, max, RMSE, etc.)

#### Details
- Computes mean, std, max displacement, and error metrics if reference is provided.
- RMSE is calculated as:
  $$
  \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left\| \mathbf{d}_i - \mathbf{d}_i^{\text{ref}} \right\|^2 }
  $$

#### Example
```python
from templatematchingpy.utils.image_utils import calculate_alignment_quality
metrics = calculate_alignment_quality(disp, true_disp)
print(metrics['rmse'])
```

---

See also: [Core Functions](core-functions.md)
