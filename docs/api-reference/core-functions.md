# Core Functions Reference

### register_stack(image_stack, bbox, reference_slice=0, config=None):
Aligns a stack of images using template matching.

#### Arguments
- **image_stack** (`np.ndarray`): 3D numpy array (slices, height, width)
- **bbox** (`Tuple[int, int, int, int]`): (x, y, w, h) tuple defining the template region
- **reference_slice** (`int`, default 0): Index of the reference slice
- **config** (`AlignmentConfig`, optional): Alignment configuration

#### Returns
- **aligned_stack** (`np.ndarray`): 3D numpy array of aligned images
- **displacements** (`List[Tuple[float, float]]`): List of (dx, dy) tuples for each slice

#### Raises
- `ValueError`: If `image_stack` is not 3D or `bbox` is invalid
- `IndexError`: If `reference_slice` is out of bounds

#### Example
```python
aligned, displacements = register_stack(stack, bbox)
```

---

See also: [StackAligner](stack-aligner.md), [AlignmentConfig](configuration.md)
