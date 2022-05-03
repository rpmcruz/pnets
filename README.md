# pnets

These are a set of evolving utilities for PyTorch to make it easy to do research on PointNet and similar networks. Not all code is mine; such code is acknowledged accordingly.

Checkout `src/train.py` for an example of how to use it.

There are other interesting packages like [Torch Points 3D](https://github.com/torch-points3d/torch-points3d), but we were having problems installing it.

## Structure

The package is made of the following modules:

* `aug`: augmentation methods
* `data`: datasets that we come with
* `plot`: plotting routines
* `pointnet`: a vanilla PointNet implementation (classifier and segmentation)

## Install

```
pip3 install git+https://github.com/rpmcruz/pnets.git
```
