# TemplateMatchingPy

Welcome to TemplateMatchingPy! This Python library is a port of the Template Matching Fiji/ImageJ plug-in developed by [Qingzong Tseng](https://github.com/qztseng/imagej_plugins). It provides fast, robust image stack alignment using template matching and OpenCV. It is designed for simple, high-throughput registration tasks where speed and ease of use are critical.

## Key Features
- Fast translation-based image stack registration
- Sub-pixel accuracy with Gaussian peak fitting
- Simple API for both beginners and advanced users
- Highly configurable alignment and matching options

## Who Should Use This?
- Researchers and engineers needing fast, simple translation alignment (e.g. Time-lapse imaging drift)
- Not intended for complex registration, (e. g. non-rigid, scaled-rotation, affine, bilinear...). For such purposes, we encorage users to try [PyStackReg](https://pypi.org/project/pystackreg/)

## Quick Start
- [Installation Guide](getting-started/installation.md)
- [5-Minute Quick Start](getting-started/quick-start.md)
- [Examples](examples/basic-examples.md)

---
For full API reference, see [API Reference](api-reference/core-functions.md).

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-EUPL%201.2-blue.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)](https://opencv.org/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/yourusername/TemplateMatchingPy)

**TemplateMatchingPy** is a Python implementation of the popular ImageJ/Fiji template matching and stack alignment plugins originally developed by [Qingzong Tseng](https://sites.google.com/site/qingzongtseng/imagejplugins), providing a programmatic, GUI-free interface for template matching and image stack alignment with sub-pixel precision designed specifically for microscopy workflows. Key features include multiple OpenCV correlation methods (TM_SQDIFF, TM_CCORR, TM_CCOEFF variants), sub-pixel precision through Gaussian peak fitting for enhanced alignment accuracy, and flexible configuration options including customizable search areas, interpolation methods, and precision settings.

This registration package is limited to Translation operations (Movements in the X-Y axis), which makes it suitable for registering time-lapses where the main image is drifting. It helps stabilising the image across time-frames. Below you can see a demostration of the capabilities of this package:

![Template Matching Alignment Demonstration](../images/comparison.gif)
*Before and after alignment comparison showing drift correction in a microscopy time-lapse sequence. The left panel shows the original drifting images, while the right panel demonstrates the stabilized result after template matching alignment.*


## Installation

```bash
pip install git+https://github.com/phisanti/TemplateMatchingPy.git
```
Or you can also build form source:
```bash
git clone https://github.com/phisanti/TemplateMatchingPy.git
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
## Configuration Options

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

## License

This project is licensed under the European Union Public Licence v. 1.2 (EUPL-1.2) - see the [LICENSE](https://github.com/phisanti/TemplateMatchingPy/blob/master/LICENSE) file for details.

## Acknowledgments

- **Qingzong Tseng**: Original ImageJ Template Matching plugin author
- **Laurent Thomas & Jochen Gehrig**: Multi-Template Matching ImageJ plugin  
- **ImageJ/Fiji Community**: Foundational image analysis tools
- **OpenCV Contributors**: Computer vision library

## Citation

If you use TemplateMatchingPy in your research, please cite:

```bibtex
@software{templatematchingpy,
  title={TemplateMatchingPy: Python implementation of ImageJ template matching and stack alignment},
  author={TemplateMatchingPy Santiago Cano Muniz},
  year={2024},
  url={https://github.com/phisanti/TemplateMatchingPy}
}
```

This implementation is based on the template matching methods described in the original research:

-  Tseng, Q. et al. A new micropatterning method of soft substrates reveals that different tumorigenic signals can promote or reduce cell contraction levels. *Lab on a Chip* 11, 2231 (2011).
-  Tseng, Q. et al. Spatial Organization of the Extracellular Matrix Regulates Cell–cell Junction Positioning. *PNAS* (2012). [doi:10.1073/pnas.1106377109](https://doi.org/10.1073/pnas.1106377109)
-  Tseng, Qingzong. 2011. "Study of multicellular architecture with controlled microenvironment". Ph.D. dissertation, Université de Grenoble. [http://tel.archives-ouvertes.fr/tel-00622264](http://tel.archives-ouvertes.fr/tel-00622264)
- Thomas, L. & Gehrig, J. Multi-template matching: a versatile tool for object-localization in microscopy images. *BMC Bioinformatics* 21, 44 (2020). [https://doi.org/10.1186/s12859-020-3363-7](https://doi.org/10.1186/s12859-020-3363-7)


## Related Projects

- [ImageJ](https://imagej.nih.gov/ij/): Java-based image processing
- [Fiji](https://fiji.sc/): ImageJ distribution with plugins
- [scikit-image](https://scikit-image.org/): Python image processing library
- [OpenCV](https://opencv.org/): Computer vision library
- [Qingzong Tseng github repo](https://github.com/qztseng/imagej_plugins)
- [Laurent Thomas github repo](https://github.com/multi-template-matching/MultiTemplateMatching-Fiji/tree/master)

---

**TemplateMatchingPy** - Bringing ImageJ template matching to Python workflows 🐍🔬
