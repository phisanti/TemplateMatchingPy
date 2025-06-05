# API Reference

## Core Functions

### register_stack

```python
register_stack(
    image_stack: np.ndarray,
    bbox: Tuple[int, int, int, int],
    reference_slice: int = 0,
    config: Optional[AlignmentConfig] = None
) -> Tuple[np.ndarray, List[Tuple[float, float]]]
```

Register an image stack using template matching.

**Parameters:**
- `image_stack`: 3D numpy array (slices, height, width)
- `bbox`: Template bounding box (x, y, width, height)
- `reference_slice`: Index of the reference slice (default: 0)  
- `config`: Alignment configuration (default: AlignmentConfig())

**Returns:**
- `aligned_stack`: 3D array of aligned images
- `displacements`: List of (dx, dy) displacements for each slice

**Raises:**
- `ValueError`: If inputs are invalid
- `IndexError`: If reference_slice is out of bounds

## Configuration Classes

### AlignmentConfig

```python
@dataclass
class AlignmentConfig:
    method: int = 5
    search_area: int = 0
    subpixel: bool = True
    interpolation: int = cv2.INTER_LINEAR
```

Configuration for template matching and stack alignment.

**Attributes:**
- `method`: OpenCV template matching method (0-5)
  - 0: TM_SQDIFF
  - 1: TM_SQDIFF_NORMED
  - 2: TM_CCORR
  - 3: TM_CCORR_NORMED
  - 4: TM_CCOEFF
  - 5: TM_CCOEFF_NORMED (default, recommended)
- `search_area`: Pixels around ROI to search (0 = entire image)
- `subpixel`: Enable sub-pixel registration using Gaussian fitting
- `interpolation`: OpenCV interpolation method for image warping

## Core Classes

### TemplateMatchingEngine

Core template matching functionality using OpenCV correlation methods.

#### Methods

##### match_template

```python
match_template(
    source: np.ndarray,
    template: np.ndarray, 
    method: int
) -> np.ndarray
```

Perform template matching using OpenCV.

##### find_peak

```python
find_peak(
    correlation_map: np.ndarray,
    method: int
) -> Tuple[int, int]
```

Find the peak location in a correlation map.

##### gaussian_peak_fit

```python
gaussian_peak_fit(
    correlation_map: np.ndarray,
    x: int,
    y: int
) -> Tuple[float, float]
```

Perform sub-pixel peak refinement using Gaussian fitting.

### StackAligner

Main stack alignment class for registering image sequences.

**Attributes:**
- `config`: AlignmentConfig instance
- `matcher`: TemplateMatchingEngine instance
- `displacements`: List of (dx, dy) displacements for each slice
- `translation_matrices`: Array of 3x3 transformation matrices (shape: [nframes, 3, 3])
- `is_registered`: Boolean flag indicating if registration has been performed

#### Methods

##### align_slice

```python
align_slice(
    source: np.ndarray,
    template: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> Tuple[float, float]
```

Align a single image slice to a template.

##### apply_translation

```python
apply_translation(
    image: np.ndarray,
    matrix: np.ndarray,
    **kwargs: Optional[dict]
) -> np.ndarray
```

Apply translation to an image using affine transformation.

**Parameters:**
- `image`: Image to be translated
- `matrix`: 3x3 transformation matrix
- `kwargs`: Additional parameters for cv2.warpAffine

**Returns:**
- Translated image with same dimensions as input

##### get_alignment

```python
get_alignment(data_type: str) -> Union[List[Tuple[float, float]], np.ndarray]
```

Retrieve stored displacements and translation matrices.

**Parameters:**
- `data_type`: Type of data to retrieve ('alignment' or 'translation_mat')

**Returns:**
- If `data_type='alignment'`: List of (dx, dy) displacements for each slice
- If `data_type='translation_mat'`: Array of 3x3 transformation matrices

**Raises:**
- `RuntimeError`: If no registration has been performed
- `ValueError`: If data_type is not 'alignment' or 'translation_mat'

##### register_stack

```python
register_stack(
    image_stack: np.ndarray,
    bbox: Tuple[int, int, int, int],
    reference_slice: int = 0
) -> np.ndarray
```

Register image stack and store alignment parameters.

**Parameters:**
- `image_stack`: 3D numpy array (slices, height, width)
- `bbox`: Template bounding box (x, y, width, height)
- `reference_slice`: Index of reference slice

**Returns:**
- `aligned_stack`: Registered image stack

**Side Effects:**
- Stores displacements in `self.displacements`
- Stores translation matrices in `self.translation_matrices`
- Sets `self.is_registered = True`

##### transform_stack

```python
transform_stack(image_stack: np.ndarray) -> np.ndarray
```

Apply stored translation matrices to a new image stack.

**Parameters:**
- `image_stack`: 3D numpy array to be transformed

**Returns:**
- `transformed_stack`: Stack with stored transformations applied

**Raises:**
- `RuntimeError`: If no registration has been performed
- `ValueError`: If stack dimensions don't match stored parameters

## Utility Functions

### Image Processing

#### validate_image_stack

```python
validate_image_stack(image_stack: np.ndarray) -> None
```

Validate that the input is a proper 3D image stack.

#### validate_bbox

```python
validate_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int]
) -> None
```

Validate bounding box coordinates against image dimensions.

#### normalize_image

```python
normalize_image(
    image: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray
```

Normalize image to [0, 1] range and convert to specified dtype.

#### extract_template

```python
extract_template(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray
```

Extract template region from image using bounding box.

### File I/O

#### save_image_stack

```python
save_image_stack(image_stack: np.ndarray, filename: str) -> None
```

Save image stack to a TIFF file.

#### load_image_stack

```python
load_image_stack(filename: str) -> np.ndarray
```

Load image stack from a TIFF file.

### Testing and Analysis

#### create_test_image_stack

```python
create_test_image_stack(
    n_slices: int = 10,
    height: int = 256,
    width: int = 256,
    noise_level: float = 0.1,
    translation_range: float = 5.0,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, List[Tuple[float, float]]]
```

Create a synthetic image stack for testing with known translations.

#### calculate_alignment_quality

```python
calculate_alignment_quality(
    displacements: List[Tuple[float, float]],
    reference_displacements: Optional[List[Tuple[float, float]]] = None
) -> dict
```

Calculate quality metrics for stack alignment.

## Constants

### MATCHING_METHODS

Mapping of method indices to OpenCV constants:

```python
MATCHING_METHODS = {
    0: cv2.TM_SQDIFF,
    1: cv2.TM_SQDIFF_NORMED,
    2: cv2.TM_CCORR,
    3: cv2.TM_CCORR_NORMED,
    4: cv2.TM_CCOEFF,
    5: cv2.TM_CCOEFF_NORMED
}
```

### MIN_VALUE_METHODS

Methods that expect minimum value (others expect maximum):

```python
MIN_VALUE_METHODS = {0, 1}  # TM_SQDIFF, TM_SQDIFF_NORMED
```

### EPS

Small constant to avoid division by zero: `1e-10`

## Usage Examples

### Basic Usage

```python
from templatematchingpy import register_stack, AlignmentConfig

# Create configuration
config = AlignmentConfig(method=5, subpixel=True)

# Align stack
aligned_stack, displacements = register_stack(
    image_stack=your_stack,
    bbox=(100, 100, 64, 64),
    reference_slice=0,
    config=config
)
```

### Advanced Configuration with StackAligner

```python
from templatematchingpy.core.stack_alignment import StackAligner
from templatematchingpy import AlignmentConfig

# Advanced configuration with search area
config = AlignmentConfig(
    method=5,                    # TM_CCOEFF_NORMED
    search_area=20,              # Limit search to 20 pixels around template
    subpixel=True,               # Enable sub-pixel precision
    interpolation=cv2.INTER_CUBIC # Use cubic interpolation
)

# Create aligner and register stack
aligner = StackAligner(config)
aligned_stack = aligner.register_stack(your_stack, bbox=(100, 100, 64, 64))

# Get alignment data
displacements = aligner.get_alignment('alignment')
translation_matrices = aligner.get_alignment('translation_mat')

# Apply same transformations to another stack
new_aligned_stack = aligner.transform_stack(another_stack)
```

### Quality Assessment

```python
from templatematchingpy import calculate_alignment_quality

# Calculate quality metrics
quality = calculate_alignment_quality(displacements)

print(f"Mean displacement: {quality['mean_displacement']:.3f} pixels")
print(f"Max displacement: {quality['max_displacement']:.3f} pixels")
print(f"Displacement std: {quality['std_displacement']:.3f} pixels")
```
