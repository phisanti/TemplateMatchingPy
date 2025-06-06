# StackAligner Class Reference

## Class Signature
```python
class StackAligner:
    def __init__(self, config: Optional[AlignmentConfig] = None):
        ...
```

---

## Methods

### __init__(config: Optional[AlignmentConfig] = None)
Initializes the stack aligner with a given configuration.

#### Arguments
- **config** (`AlignmentConfig`, optional): Alignment configuration. If `None`, uses default settings.

#### Example
```python
aligner = StackAligner()
aligner_custom = StackAligner(config=AlignmentConfig(method=5, subpixel=True))
```

---

### align_slice(source, template, bbox)
Align a single image slice to a template using template matching.

#### Arguments
- **source** (`np.ndarray`): Source image to align (2D array)
- **template** (`np.ndarray`): Reference template (2D array)
- **bbox** (`Tuple[int, int, int, int]`): Bounding box `(x, y, width, height)`

#### Returns
- **(dx, dy)** (`Tuple[float, float]`): Displacement needed to align source to template

#### Raises
- `ValueError`: If bounding box is invalid or template/source sizes are incompatible

#### Details
- Optionally restricts search area for speed
- Uses sub-pixel refinement if enabled in config

#### Example
```python
dx, dy = aligner.align_slice(img, template, (x, y, w, h))
```

---

### register_stack(image_stack, bbox, reference_slice=0, reference_type="static")
Register an image stack and store alignment parameters.

#### Arguments
- **image_stack** (`np.ndarray`): 3D array (slices, height, width)
- **bbox** (`Tuple[int, int, int, int]`): Template bounding box
- **reference_slice** (`int`, default 0): Index of reference slice (static) or negative offset (dynamic)
- **reference_type** (`str`, default 'static'): 'static' uses fixed reference, 'dynamic' uses previous slices

#### Returns
- **aligned_stack** (`np.ndarray`): Registered image stack

#### Raises
- `ValueError`, `IndexError`: For invalid input or reference

#### Details
- Stores displacements and translation matrices for later use
- Supports both static and dynamic reference modes

#### Example
```python
aligned = aligner.register_stack(stack, bbox, reference_slice=0)
```

---

### apply_translation(image, matrix, **kwargs)
Apply a translation to an image using a 3x3 transformation matrix.

#### Arguments
- **image** (`np.ndarray`): Image to translate
- **matrix** (`np.ndarray`): 3x3 transformation matrix
- **kwargs**: Additional parameters for `cv2.warpAffine`

#### Returns
- **translated_image** (`np.ndarray`): Translated image

#### Example
```python
translated = aligner.apply_translation(img, matrix)
```

---

### transform_stack(image_stack)
Apply stored translation matrices to a new image stack.

#### Arguments
- **image_stack** (`np.ndarray`): 3D array to transform

#### Returns
- **transformed_stack** (`np.ndarray`): Stack with stored transformations applied

#### Raises
- `RuntimeError`: If no registration has been performed
- `ValueError`: If stack dimensions do not match registration

#### Example
```python
transformed = aligner.transform_stack(new_stack)
```

---

### get_alignment(data_type)
Retrieve stored displacements or translation matrices.

#### Arguments
- **data_type** (`str`): 'alignment' for displacements, 'translation_mat' for matrices

#### Returns
- `List[Tuple[float, float]]` or `np.ndarray`: Displacements or translation matrices

#### Raises
- `RuntimeError`: If no registration has been performed
- `ValueError`: If data_type is invalid

#### Example
```python
displacements = aligner.get_alignment('alignment')
matrices = aligner.get_alignment('translation_mat')
```

---

## State Variables
- **is_registered** (`bool`): True if registration has been performed
- **displacements** (`List[Tuple[float, float]]`): Displacement for each slice
- **translation_matrices** (`np.ndarray`): 3x3 matrices for each slice

---

## Mathematical Notes
- The translation matrix for each slice is:
  $$
  T = \begin{bmatrix} 1 & 0 & dx \\ 0 & 1 & dy \\ 0 & 0 & 1 \end{bmatrix}
  $$
- Displacements $(dx, dy)$ are computed by maximizing (or minimizing) the template matching score, optionally refined to sub-pixel accuracy.

---

## Example
```python
from templatematchingpy.core.stack_alignment import StackAligner
from templatematchingpy.core.config import AlignmentConfig

aligner = StackAligner(AlignmentConfig(method=5, subpixel=True))
aligned = aligner.register_stack(stack, bbox)
displacements = aligner.get_alignment('alignment')
```

---

See also: [Core Functions](core-functions.md), [Configuration](configuration.md)
