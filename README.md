# Project of lab course Vision-based Navigation SS19 at TUM
Implementation of Direct Visual Odometry for RGB-D Images

Dataset: https://vision.in.tum.de/data/datasets/rgbd-dataset/download

## Tasks:
- Estimate the relative pose via Direct Image Alignment
- Implement Gauss-Newton (or LM) manually
- Frame-to-frame or frame-to-keyframe
- Different image warping strategies
- Coarse-to-fine to improve convergence
- Robust-norm to handle outliers

## Dependencies
- Eigen (3.3.7 tested)
- Sophus (1.0.0 tested)
- Pangolin (0.5 tested)
- OpenCV (4.1.0 tested)

## References
- Robust Odometry Estimation for RGB-D Cameras (C. Kerl, J. Sturm and D. Cremers), In International Conference on Robotics and Automation (ICRA), 2013.
(https://vision.in.tum.de/_media/spezial/bib/kerl13icra.pdf)
- Equivalence and efficiency of image alignment algorithms (Baker, Simon, and Iain Matthews), In IEEE Computer Society Conference on Computer Vision and Pattern Recognition. Vol. 1. IEEE Computer Society; 1999, 2001.
(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.20&rep=rep1&type=pdf)
- Dense Visual Odometry
(https://github.com/tum-vision/dvo)
