# pnets

These are a set of evolving utilities for PyTorch to make it easy to do research on PointNet and similar networks. Not all code is mine; such code is acknowledged accordingly.

There are other interesting packages like [Torch Points 3D](https://github.com/torch-points3d/torch-points3d), but we had some problems installing it and decided to build our own package for fun and profit. :-)

## Structure

The package is made of the following modules:

* [`aug`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/aug.html): augmentation methods
* [`data`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/data.html): datasets that we come with
* [`plot`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/plot.html): plotting routines
* [`pointnet`](http://htmlpreview.github.io/?https://github.com/rpmcruz/objdetect/blob/main/html/pointnet.html): a vanilla PointNet implementation (for classification and segmentation)

You can have a look at some examples under the `src` folder.

## Install

```
pip3 install git+https://github.com/rpmcruz/pnets.git
```
