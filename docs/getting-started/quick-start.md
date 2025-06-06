# Quick Start

## Purpose
A 5-minute tutorial to get you up and running with TemplateMatchingPy.

## Example: Align a Synthetic Image Stack
```python
import numpy as np
from templatematchingpy import register_stack

# Create a synthetic stack (replace with your data)
stack = np.random.randint(0, 255, (10, 128, 128), dtype=np.uint8)
bbox = (32, 32, 64, 64)  # x, y, width, height
aligned, displacements = register_stack(stack, bbox)
```

- `aligned` is the registered stack
- `displacements` is a list of (dx, dy) for each slice

## Visualize Results
```python
import matplotlib.pyplot as plt
plt.imshow(aligned[0])
plt.title('Aligned Reference Slice')
plt.show()
```

See [Basic Examples](../examples/basic-examples.md) for more.
