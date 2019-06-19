#ifndef DVO_DIRECTODOMETRY_H
#define DVO_DIRECTODOMETRY_H

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
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

  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> gradx_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> grady_Pyramid;

  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>
      intr_Pyramid;

  // Robust weight estimation
  static constexpr float INIT_SIGMA = 5.0;
  static constexpr float DEFAULT_DOF = 5.0;

  void makePyramid();

  void calcGradient(const cv::Mat &img, cv::Mat &grad_x, cv::Mat &grad_y);

  void calcResiduals(const Sophus::SE3f &xi, const int level,
                     Eigen::VectorXf &residuals);

  void showError(const Sophus::SE3f &xi, const int level);

  void calcJacobian(const Sophus::SE3f &xi, const int level,
                    Eigen::MatrixXf &J);

  void weighting(const Eigen::VectorXf &residuals, Eigen::VectorXf &weights);

public:
  DirectOdometry(const cv::Mat &pImg, const cv::Mat &pDep, const cv::Mat &cImg,
                 const Eigen::Vector4f &intr, const float FACTOR) {

    pImg.convertTo(this->pImg, CV_32FC1, 1.0 / 255.0);
    cImg.convertTo(this->cImg, CV_32FC1, 1.0 / 255.0);
    pDep.convertTo(this->pDep, CV_32FC1, 1.0 / FACTOR);
    this->intr = intr;
  }

  Sophus::SE3f optimize();
};

#endif // DVO_DIRECTODOMETRY_H
