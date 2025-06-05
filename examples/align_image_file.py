import cv2
import numpy as np
from templatematchingpy import register_stack, AlignmentConfig

# Load multi-page TIFF stack
ret, images = cv2.imreadmulti(
    "./examples/data/example_image_stack.tiff", flags=cv2.IMREAD_GRAYSCALE
)

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
