# Randomized affine transformations
### Requirements:

  1) torchvision 0.8.2
  2) torch 1.7.1

Multiple transformations are implemented in this code:
  - Horizontal flip
  - Scale
  - Shear
  - Rotate
  - Translate


It keeps tracking the original bounding box coordinates of groundtruth and generates the new bounding box after transforming. Then applying random occlusion.
