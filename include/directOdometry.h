#ifndef DVO_DIRECTODOMETRY_H
#define DVO_DIRECTODOMETRY_H

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <utils.h>

class DirectOdometry {
private:
  // Parameters
  static const int NUM_PYRAMID = 5;
  static const int NUM_GNITERS = 20;

  // Image and camera matrix.
  cv::Mat pImg, pDep, cImg;
  Intrinsics intr;
  int W, H;

  // Image and camera matrix pyramids.
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> pImg_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> pDep_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> cImg_Pyramid;

  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> gradx_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> grady_Pyramid;

  std::vector<Intrinsics, Eigen::aligned_allocator<Intrinsics>> intr_Pyramid;

  // Robust weight estimation
  static constexpr float INIT_SIGMA = 5.0;
  static constexpr float DEFAULT_DOF = 5.0;

  void makePyramid();

  void calcGradient(const cv::Mat &img, cv::Mat &grad_x, cv::Mat &grad_y);

  void calcResiduals(const Transform &xi, int level,
                     Eigen::VectorXf &residuals);

  void calcFinalRes(const Transform &xi);

  void showError(const Transform &xi, int level);

  void calcJacobian(const Transform &xi, int level, Eigen::MatrixXf &J);

  void weighting(const Eigen::VectorXf &residuals, Eigen::VectorXf &weights);

public:
  cv::Mat finalResidual;

  DirectOdometry(const cv::Mat &pImg, const cv::Mat &pDep, const cv::Mat &cImg,
                 const Intrinsics &intr, float FACTOR) {

    pImg.convertTo(this->pImg, CV_32FC1, 1.0 / 255.0);
    cImg.convertTo(this->cImg, CV_32FC1, 1.0 / 255.0);
    pDep.convertTo(this->pDep, CV_32FC1, 1.0 / FACTOR);
    this->intr = intr;
    W = pImg.cols;
    H = pImg.rows;
    finalResidual = cv::Mat::zeros(H, W, CV_32FC1);
  }

  Transform
  optimize(Transform init_xi = Transform(Eigen::Matrix4f::Identity()));
};

#endif // DVO_DIRECTODOMETRY_H
