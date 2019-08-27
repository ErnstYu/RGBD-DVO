#include <directOdometry.h>

float calcError(Eigen::VectorXf &residuals) {
  float num = 0.0, err = 0.0;

  for (int i = 0; i < residuals.size(); ++i)
    if (!std::isnan(residuals(i))) {
      err += residuals(i) * residuals(i);
      num += 1.0;
    }
  err = err / num;
  return err;
}

void DirectOdometry::calcResiduals(const Transform &xi, const int level,
                                   Eigen::VectorXf &residuals) {

  Intrinsics intr_level = pre.intr_Pyramid[level];
  cv::Mat cImg_level = cur.gray_Pyramid[level];
  cv::Mat pImg_level = pre.gray_Pyramid[level];
  cv::Mat pDep_level = pre.depth_Pyramid[level];

  int w = pImg_level.cols;
  int h = pImg_level.rows;

  // camera intrinsics
  float fx = intr_level(0);
  float fy = intr_level(1);
  float cx = intr_level(2);
  float cy = intr_level(3);

  Mat3 R = xi.rotationMatrix();
  Vec3 t = xi.translation();

  float *ptr_pImg = (float *)pImg_level.data;
  float *ptr_pDep = (float *)pDep_level.data;
  float *ptr_cImg = (float *)cImg_level.data;

  residuals = Eigen::VectorXf::Ones(w * h) * nan("2");

  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u) {
      int pos = v * w + u;
      if (ptr_pDep[pos] < 1e-6)
        continue;

      Vec3 pt3d(((float)u - cx) / fx * ptr_pDep[pos],
                ((float)v - cy) / fy * ptr_pDep[pos], ptr_pDep[pos]);
      pt3d = R * pt3d + t;

      if (pt3d[2] > 1e-6) {
        float x = fx * pt3d[0] / pt3d[2] + cx;
        float y = fy * pt3d[1] / pt3d[2] + cy;

        float color_warped = interpolate(ptr_cImg, x, y, w, h);
        if (!std::isnan(color_warped))
          residuals[pos] = color_warped - ptr_pImg[pos];
      }
    }
}

void DirectOdometry::weighting(const Eigen::VectorXf &residuals,
                               Eigen::VectorXf &weights) {
  int n = residuals.size();
  float lambda_prev, lambda = INIT_SIGMA * INIT_SIGMA;
  float num = 0.0;
  weights = Eigen::VectorXf::Zero(n);
  do {
    lambda_prev = lambda;
    lambda = 0.0;
    num = 0.0;
    for (int i = 0; i < n; ++i) {
      float res = residuals(i);

      if (!std::isnan(res)) {
        num += 1.0;
        lambda +=
            res * res *
            ((DEFAULT_DOF + 1.0) / (DEFAULT_DOF + res * res / lambda_prev));
      }
    }
    lambda = lambda / num;
  } while (std::abs(lambda - lambda_prev) > 1e-6);

  for (int i = 0; i < n; ++i) {
    float res = residuals(i);
    if (!std::isnan(res))
      weights(i) = ((DEFAULT_DOF + 1.0) / (DEFAULT_DOF + lambda * res * res));
  }
}

void DirectOdometry::calcFinalRes(const Transform &xi) {
  Eigen::VectorXf res;
  calcResiduals(xi, 0, res);

  float *ptr_res = (float *)finalResidual.data;

  for (int v = 0; v < pre.H; ++v) {
    for (int u = 0; u < pre.W; ++u) {
      int pos = v * pre.W + u;
      if (!std::isnan(res[pos]))
        ptr_res[pos] = abs(res[pos] * 255.0);
    }
  }
}

void DirectOdometry::showError(const Transform &xi, const int level) {
  static int itr = 0;
  Eigen::VectorXf res;
  calcResiduals(xi, 0, res);

  cv::Mat err = cv::Mat::zeros(pre.H, pre.W, CV_32FC1);
  float *ptr_err = (float *)err.data;

  for (int v = 0; v < pre.H; ++v) {
    for (int u = 0; u < pre.W; ++u) {
      int pos = v * pre.W + u;
      if (!std::isnan(res[pos]))
        ptr_err[pos] = abs(res[pos] * 255.0);
    }
  }
  cv::imwrite(std::to_string(itr) + '_' + std::to_string(level) + ".png", err);
  itr += 1;
}

void DirectOdometry::calcJacobian(const Transform &xi, const int level,
                                  Eigen::MatrixXf &J) {

  Intrinsics intr_level = pre.intr_Pyramid[level];
  cv::Mat cImg_level = cur.gray_Pyramid[level];
  cv::Mat pDep_level = pre.depth_Pyramid[level];
  cv::Mat gradx_level = cur.gradx_Pyramid[level];
  cv::Mat grady_level = cur.grady_Pyramid[level];

  // Camera intrinsics
  float fx = intr_level(0);
  float fy = intr_level(1);
  float cx = intr_level(2);
  float cy = intr_level(3);

  // Width and Height
  int w = cImg_level.cols;
  int h = cImg_level.rows;

  float *ptr_gradx = (float *)gradx_level.data;
  float *ptr_grady = (float *)grady_level.data;
  float *ptr_pDep = (float *)pDep_level.data;

  // RationMatrix and t
  Mat3 R = xi.rotationMatrix();
  Vec3 t = xi.translation();

  // Jacobian
  J = Eigen::MatrixXf::Zero(w * h, 6);
  Eigen::MatrixXf JI(1, 2), Jw(2, 6);

  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      int pos = v * w + u;
      if (ptr_pDep[pos] < 1e-6)
        continue;

      Vec3 pt3d(((float)u - cx) / fx * ptr_pDep[pos],
                ((float)v - cy) / fy * ptr_pDep[pos], ptr_pDep[pos]);
      pt3d = R * pt3d + t;

      float X = pt3d[0];
      float Y = pt3d[1];
      float Z = pt3d[2];

      if (Z < 1e-6)
        continue;

      float x = fx * X / Z + cx;
      float y = fy * Y / Z + cy;

      if (0 <= x && x < w && 0 <= y && y < h) {
        Jw << fx / Z, 0, -fx * X / (Z * Z), -fx * (X * Y) / (Z * Z),
            fx * (1 + (X * X) / (Z * Z)), -fx * Y / Z, 0, fy / Z,
            -fy * Y / (Z * Z), -fy * (1 + (Y * Y) / (Z * Z)),
            fy * X * Y / (Z * Z), fy * X / Z;

        JI(0, 0) = interpolate(ptr_gradx, x, y, w, h);
        JI(0, 1) = interpolate(ptr_grady, x, y, w, h);
        J.row(pos) = JI * Jw;
      }
    }
  }
}

Transform DirectOdometry::optimize(Transform init_xi) {

  Transform xi(init_xi);
  Eigen::VectorXf xi_inc(6);
  Eigen::VectorXf residuals, weights;
  Eigen::MatrixXf J;

  for (int level = NUM_PYRAMID - 1; level >= 0; --level) {
    float error_prev = std::numeric_limits<float>::max();
    for (int itr = 0; itr < NUM_GNITERS; ++itr) {

      // showError(xi, level);

      // compute residuals
      calcResiduals(xi, level, residuals);

      float error = calcError(residuals);

      // // break at convergence and (possibly) reject last increment
      if (error > error_prev) {
        xi = Transform::exp(xi_inc).inverse() * xi;
        break;
      }
      if (error / error_prev > (1 - 3e-3))
        break;
      // std::cout << itr << ": " << error << std::endl;
      error_prev = error;

      // r = W * r
      weighting(residuals, weights);
      // residuals = residuals.cwiseProduct(weights);
      for (int i = 0; i < residuals.size(); ++i)
        residuals(i) = isnan(residuals(i)) ? 0 : weights[i] * residuals(i);

      calcJacobian(xi, level, J);
      Eigen::MatrixXf Jt = J.transpose();

      // J = W * J
      for (int i = 0; i < J.rows(); ++i)
        J.row(i) = weights[i] * J.row(i);

      // Gauss-Newton method
      // Jt * (W * J) * xi_inc = - Jt * (W * r)
      xi_inc = (Jt * J).ldlt().solve(-Jt * residuals);

      // Levenberg-Marquardt method
      // (Jt * (W * J) + lambda * diag(Jt * (W * J))) * xi_inc = - Jt * (W * r)
      //
      // xi_inc = (Jt * J + lambda * (Jt * J).diagonal().asDiagonal())
      //          .ldlt().solve(-Jt * residuals);

      xi = Transform::exp(xi_inc) * xi;
    }
  }
  calcFinalRes(xi);

  return xi;
}
