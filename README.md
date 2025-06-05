# TemplateMatchingPy
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-EUPL%201.2-blue.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)](https://opencv.org/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/yourusername/TemplateMatchingPy)

**TemplateMatchingPy** is a Python implementation of the popular ImageJ/Fiji template matching and stack alignment plugins originally developed by Qingzong Tseng, providing a programmatic, GUI-free interface for template matching and image stack alignment with sub-pixel precision designed specifically for microscopy workflows. Key features include multiple OpenCV correlation methods (TM_SQDIFF, TM_CCORR, TM_CCOEFF variants), sub-pixel precision through Gaussian peak fitting for enhanced alignment accuracy, and flexible configuration options including customizable search areas, interpolation methods, and precision settings.

This registration package is limited to Translation operations (Movements in the X-Y axis), which makes it suitable for registering time-lapses where the main image is drifting. It helps stabilising the image across time-frames. Below you can see a demostration of the capabilities of this package:

![Template Matching Alignment Demonstration](./examples/data/comparison.gif)

*Before and after alignment comparison showing drift correction in a microscopy time-lapse sequence. The left panel shows the original drifting images, while the right panel demonstrates the stabilized result after template matching alignment.*


## 📦 Installation

```bash
pip install git+https://github.com/yourusername/TemplateMatchingPy.git
```
Or you can also build form source:
```bash
git clone https://github.com/yourusername/TemplateMatchingPy.git
cd TemplateMatchingPy
pip install -e .
```

### Dependencies

- Python ≥ 3.7
- NumPy ≥ 1.19.0  
- OpenCV ≥ 4.5.0

## Basic Usage

```python
import numpy as np
from templatematchingpy import (
    register_stack,
    AlignmentConfig,
    create_test_image_stack,
    calculate_alignment_quality,
)

# Create test image stack (or load your own)
image_stack, true_displacements = create_test_image_stack(
    n_slices=8, height=256, width=256, translation_range=5.0, noise_level=0.1
)

# Define template region (x, y, width, height)
bbox = (100, 100, 64, 64)

# Configure alignment
config = AlignmentConfig(method=5, subpixel=True)

# Perform alignment
aligned_stack, displacements = register_stack(
    image_stack=image_stack,
    bbox=bbox,
    reference_slice=0,
    config=config
)

print(f"Aligned {len(displacements)} slices")
print(f"Mean displacement: {np.mean([np.sqrt(dx**2 + dy**2) for dx, dy in displacements]):.2f} pixels")
```

### Working with Files

```python
import cv2
import numpy as np
from templatematchingpy import register_stack, AlignmentConfig

# Load multi-page TIFF stack
ret, images = cv2.imreadmulti("./examples/data/example_image_stack.tiff", flags=cv2.IMREAD_GRAYSCALE)

if not ret:
    raise ValueError("Could not load TIFF stack")

# Convert list of images to 3D numpy array [frames, height, width]
image_stack = np.array(images, dtype=np.float32)

# Normalize to [0, 1] range if needed
if image_stack.max() > 1.0:
    image_stack = image_stack / 255.0

print(f"Loaded stack with shape: {image_stack.shape}")

# Get image dimensions and calculate centered bbox
height, width = image_stack.shape[1], image_stack.shape[2]
box_width = 1200
box_height = 1200  
x = (width - box_width) // 2
y = (height - box_height) // 2

# Define template region (x, y, width, height)
bbox = (x, y, box_width, box_height)

# Configure and perform alignment
config = AlignmentConfig(method=5, subpixel=True)
aligned_stack, displacements = register_stack(
    image_stack, bbox=bbox, reference_slice=0, config=config
)

# Save aligned stack as float32 multi-page TIFF
# OpenCV requires list of individual frames for multi-page TIFF
aligned_frames = [frame.astype(np.float32) for frame in aligned_stack]
cv2.imwritemulti("aligned_stack.tiff", aligned_frames)

print(f"Alignment completed with {len(displacements)} slices")
print(f"Displacements: {displacements}")
```
## 🔧 Configuration Options

### AlignmentConfig Parameters

```python
from templatematchingpy import AlignmentConfig
import cv2

config = AlignmentConfig(
    method=5,                    # Template matching method (0-5)
    search_area=0,               # Search area in pixels (0 = full image)  
    subpixel=True,               # Enable sub-pixel precision
    interpolation=cv2.INTER_LINEAR  # Interpolation method
)
```

### Template Matching Methods

| Method | OpenCV Constant | Description | Best For |
|--------|----------------|-------------|----------|
| 0 | TM_SQDIFF | Squared Difference | High contrast templates |
| 1 | TM_SQDIFF_NORMED | Normalized Squared Difference | Robust matching |
| 2 | TM_CCORR | Cross Correlation | Bright templates |
| 3 | TM_CCORR_NORMED | Normalized Cross Correlation | Illumination invariant |
| 4 | TM_CCOEFF | Correlation Coefficient | General purpose |
| **5** | **TM_CCOEFF_NORMED** | **Normalized Correlation Coefficient** | **Recommended** |

## 📖 Documentation

- **[API Reference](docs/api_reference.md)**: Complete function and class documentation
- **[Tutorial](docs/tutorial.md)**: Step-by-step guide with real examples
- **[Examples](examples/)**: Ready-to-run example scripts
  - [`basic_usage.py`](examples/basic_usage.py): Getting started example
  - [`advanced_alignment.py`](examples/advanced_alignment.py): Advanced features and comparison
  - [`batch_processing.py`](examples/batch_processing.py): Batch processing workflow

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=templatematchingpy --cov-report=html
```

Test coverage includes:
- Unit tests for all core functions
- Integration tests with synthetic data
- Edge case and error handling tests
- Performance regression tests

## 🏁 Performance Guidelines

### Template Selection
- **Size**: 32-128 pixels square typically optimal
- **Features**: High contrast, distinctive patterns
- **Location**: Avoid edges and homogeneous regions

### Speed Optimization
```python
# For faster processing
config = AlignmentConfig(
    method=5,
    search_area=30,     # Restrict search area
    subpixel=False      # Disable sub-pixel precision
)
```

### Memory Management
```python
# For large stacks, process in chunks
def process_large_stack(large_stack, chunk_size=50):
    results = []
    for i in range(0, len(large_stack), chunk_size):
        chunk = large_stack[i:i+chunk_size]
        aligned_chunk, displacements = register_stack(chunk, bbox)
        results.append((aligned_chunk, displacements))
    return results
```

## 🎓 Scientific Background

This implementation is based on the template matching methods described in:

> **Multi-template matching: a versatile tool for object-localization in microscopy images**  
> Laurent Thomas, Jochen Gehrig  
> *BMC Bioinformatics* 21, 44 (2020)  
> https://doi.org/10.1186/s12859-020-3363-7

### Original ImageJ Plugin

This Python implementation replicates the functionality of the ImageJ/Fiji plugins:

- **Template Matching**: https://sites.google.com/site/qingzongtseng/template-matching-ij-plugin
- **Multi-Template Matching**: https://github.com/multi-template-matching/MultiTemplateMatching-Fiji

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/TemplateMatchingPy.git
cd TemplateMatchingPy
pip install -e .[dev]
```

### Code Quality

```bash
# Format code
black templatematchingpy/

# Type checking  
mypy templatematchingpy/

# Linting
flake8 templatematchingpy/
```

## 📄 License

This project is licensed under the European Union Public Licence v. 1.2 (EUPL-1.2) - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qingzong Tseng**: Original ImageJ Template Matching plugin author
- **Laurent Thomas & Jochen Gehrig**: Multi-Template Matching ImageJ plugin  
- **ImageJ/Fiji Community**: Foundational image analysis tools
- **OpenCV Contributors**: Computer vision library

## 📬 Citation

If you use TemplateMatchingPy in your research, please cite:

```bibtex
@software{templatematchingpy,
  title={TemplateMatchingPy: Python implementation of ImageJ template matching and stack alignment},
  author={TemplateMatchingPy Contributors},
  year={2024},
  url={https://github.com/yourusername/TemplateMatchingPy}
}
```

And the original research:

```bibtex
@article{thomas2020multi,
  title={Multi-template matching: a versatile tool for object-localization in microscopy images},
  author={Thomas, Laurent and Gehrig, Jochen},
  journal={BMC bioinformatics},
  volume={21},
  number={1},
  pages={1--15},
  year={2020},
  publisher={Springer}
}
```

## 🔗 Related Projects

- [ImageJ](https://imagej.nih.gov/ij/): Java-based image processing
- [Fiji](https://fiji.sc/): ImageJ distribution with plugins
- [scikit-image](https://scikit-image.org/): Python image processing library
- [OpenCV](https://opencv.org/): Computer vision library

---

**TemplateMatchingPy** - Bringing ImageJ template matching to Python workflows 🐍🔬
