#ifndef DVO_DIRECTODOMETRY_H
#define DVO_DIRECTODOMETRY_H

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <frame.h>
#include <opencv2/opencv.hpp>
#include <utils.h>

class DirectOdometry {
private:
  // Parameters
  static const int NUM_GNITERS = 20;

  // Previous and current frame
  Frame pre, cur;

  // Robust weight estimation
  static constexpr float INIT_SIGMA = 5.0;
  static constexpr float DEFAULT_DOF = 5.0;

  void calcResiduals(const Transform &xi, int level,
                     Eigen::VectorXf &residuals);

  void calcFinalRes(const Transform &xi);

  void showError(const Transform &xi, int level);

  void calcJacobian(const Transform &xi, int level, Eigen::MatrixXf &J);

  void weighting(const Eigen::VectorXf &residuals, Eigen::VectorXf &weights);

public:
  cv::Mat finalResidual;

  DirectOdometry(const Frame &prev, const Frame &curr) : pre(prev), cur(curr) {
    finalResidual = cv::Mat::zeros(pre.H, pre.W, CV_32FC1);
  }

  Transform
  optimize(Transform init_xi = Transform(Eigen::Matrix4f::Identity()));
};

#endif // DVO_DIRECTODOMETRY_H
