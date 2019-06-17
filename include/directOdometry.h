#ifndef DVO_DIRECTODOMETRY_H
#define DVO_DIRECTODOMETRY_H

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

class DirectOdometry {
private:
  // Parameters
  static const int NUM_PYRAMID = 5;
  static const int NUM_GNITERS = 20;

  // Image and camera matrix.
  cv::Mat pImg, pDep, cImg;
  Eigen::Vector4f intr;

  // Image and camera matrix pyramids.
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> pImg_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> pDep_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> cImg_Pyramid;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>
      intr_Pyramid;

  // Robust weight estimation
  static constexpr float INIT_SIGMA = 5.0;
  static constexpr float DEFAULT_DOF = 5.0;

  void makePyramid();

  void calcGradient(const cv::Mat &img, cv::Mat &gradient, int direction);

  Eigen::VectorXf calcResiduals(const Sophus::SE3f &xi, const int level);

  Eigen::MatrixXf calcJacobian(const Sophus::SE3f &xi, const int level);

  void weighting(const Eigen::VectorXf &residuals, Eigen::VectorXf &weights);

public:
  DirectOdometry(const cv::Mat &pImg, const cv::Mat &pDep, const cv::Mat &cImg,
                 const Eigen::Vector4f &intr) {

    pImg.convertTo(this->pImg, CV_32FC1, 1.0 / 255.0);
    cImg.convertTo(this->cImg, CV_32FC1, 1.0 / 255.0);
    pDep.convertTo(this->pDep, CV_32FC1);
    this->intr = intr;
  }

  Sophus::SE3f optimize();
};

#endif // DVO_DIRECTODOMETRY_H
