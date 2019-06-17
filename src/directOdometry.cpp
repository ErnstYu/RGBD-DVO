#include <directOdometry.h>

float interpolate(const float *img_ptr, float x, float y, int w, int h) {
  float val_itpltd = nan("1");
  int x0 = floor(x), y0 = floor(y);
  int x1 = x0 + 1, y1 = y0 + 1;

  float x1_weight = x - x0, y1_weight = y - y0;
  float x0_weight = 1 - x1_weight, y0_weight = 1 - y1_weight;

  // Check if warped points are in the image.
  if (x0 < 0 or x0 >= w)
    x0_weight = 0;
  if (x1 < 0 or x1 >= w)
    x1_weight = 0;
  if (y0 < 0 or y0 >= h)
    y0_weight = 0;
  if (y1 < 0 or y1 >= h)
    y1_weight = 0;
  float w00 = x0_weight * y0_weight;
  float w10 = x1_weight * y0_weight;
  float w01 = x0_weight * y1_weight;
  float w11 = x1_weight * y1_weight;

  // Compute interpolated pixel intensity.
  float sumWeights = w00 + w10 + w01 + w11;
  float total = 0;
  if (w00 > 0)
    total += img_ptr[y0 * w + x0] * w00;
  if (w01 > 0)
    total += img_ptr[y1 * w + x0] * w01;
  if (w10 > 0)
    total += img_ptr[y0 * w + x1] * w10;
  if (w11 > 0)
    total += img_ptr[y1 * w + x1] * w11;

  if (sumWeights > 0)
    val_itpltd = total / sumWeights;

  return val_itpltd;
}

cv::Mat downsampleImg(const cv::Mat &img) {

  int w = img.cols, h = img.rows;
  int w_ds = w / 2, h_ds = h / 2;

  float *input_ptr = (float *)img.ptr();
  cv::Mat img_ds = cv::Mat::zeros(h_ds, w_ds, img.type());
  float *output_ptr = (float *)img_ds.data;

  for (int y = 0; y < h_ds; y++) {
    for (int x = 0; x < w_ds; x++) {
      output_ptr[y * w_ds + x] +=
          (input_ptr[2 * y * w + 2 * x] + input_ptr[2 * y * w + 2 * x + 1] +
           input_ptr[(2 * y + 1) * w + 2 * x] +
           input_ptr[(2 * y + 1) * w + 2 * x + 1]) /
          4.0;
    }
  }

  return img_ds;
}

cv::Mat downsampleDepth(const cv::Mat &depth) {

  int w = depth.cols, h = depth.rows;
  int w_ds = w / 2, h_ds = h / 2;
  float *input_ptr = (float *)depth.ptr();
  cv::Mat depth_ds = cv::Mat::zeros(h_ds, w_ds, depth.type());
  float *output_ptr = (float *)depth_ds.data;

  for (int y = 0; y < h_ds; y++) {
    for (int x = 0; x < w_ds; x++) {
      int top_left = 2 * y * w + 2 * x;
      int top_right = top_left + 1;
      int btm_left = (2 * y + 1) * w + 2 * x;
      int btm_right = btm_left + 1;
      int count = 0;
      float total = 0.0;

      // To keep the border of 3D shape, a pixel without depth is ignored.
      if (input_ptr[top_left] != 0.0) {
        total += input_ptr[top_left];
        count++;
      }

      if (input_ptr[top_right] != 0.0) {
        total += input_ptr[top_right];
        count++;
      }

      if (input_ptr[btm_left] != 0.0) {
        total += input_ptr[btm_left];
        count++;
      }

      if (input_ptr[btm_right] != 0.0) {
        total += input_ptr[btm_right];
        count++;
      }

      if (count > 0) {
        output_ptr[y * w_ds + x] = total / (float)count;
      }
    }
  }

  return depth_ds;
}

void DirectOdometry::makePyramid() {
  intr_Pyramid.push_back(this->intr);
  pImg_Pyramid.push_back(this->pImg);
  pDep_Pyramid.push_back(this->pDep);
  cImg_Pyramid.push_back(this->cImg);

  for (int i = 1; i < NUM_PYRAMID; i++) {
    // downsample camera matrix
    this->intr_Pyramid.push_back(this->intr_Pyramid[i - 1] * 0.5);

    // downsample grayscale images
    cv::Mat pImgDown = downsampleImg(this->pImg_Pyramid[i - 1]);
    cv::Mat cImgDown = downsampleImg(this->cImg_Pyramid[i - 1]);
    this->pImg_Pyramid.push_back(pImgDown);
    this->cImg_Pyramid.push_back(cImgDown);

    // downsample depth images
    cv::Mat pDepDown = downsampleDepth(this->pDep_Pyramid[i - 1]);
    this->pDep_Pyramid.push_back(pDepDown);
  }

  return;
}

Eigen::VectorXf DirectOdometry::calcResiduals(const Sophus::SE3f &xi,
                                              const int level) {

  Eigen::VectorXf residuals;

  Eigen::Vector4f klevel = this->intr_Pyramid[level];
  cv::Mat cImg_level = this->cImg_Pyramid[level];
  cv::Mat pImg_level = this->pImg_Pyramid[level];
  cv::Mat pDep_level = this->pDep_Pyramid[level];

  int w = pImg_level.cols;
  int h = pImg_level.rows;

  // camera intrinsics
  float fx = klevel(0);
  float fy = klevel(1);
  float cx = klevel(2);
  float cy = klevel(3);

  // convert SE3 to Ration matrix and translation vector
  Eigen::Matrix3f R = xi.rotationMatrix();
  Eigen::Vector3f t = xi.translation();

  float *ptr_pImg = (float *)pImg_level.data;
  float *ptr_pDep = (float *)pDep_level.data;
  float *ptr_cImg = (float *)cImg_level.data;

  residuals.resize(w * h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int pos = y * w + x;
      float residual = 0.0;

      Eigen::Vector3f pt_3d((x + cx) / fx * ptr_pDep[pos],
                            (y + cy) / fy * ptr_pDep[pos], ptr_pDep[pos]);
      pt_3d = R * pt_3d + t;

      if (pt_3d[2] > 0.0) {
        Eigen::Vector2f pt_2d(fx * pt_3d[0] / pt_3d[2] - cx,
                              fy * pt_3d[0] / pt_3d[2] - cy);

        float color_warped = interpolate(ptr_cImg, pt_2d[0], pt_2d[1], w, h);
        if (!std::isnan(color_warped)) {
          float color_prev = ptr_pImg[pos];
          residual = color_prev - color_warped;
        }
      }

      residuals[pos] = residual;
    }
  }

  return residuals;
}

void DirectOdometry::weighting(const Eigen::VectorXf &residuals,
                               Eigen::VectorXf &weights) {
  int n = residuals.size();
  float lambda_prev, lambda = 1.0 / (INIT_SIGMA * INIT_SIGMA);
  float num = 0.0;
  weights = Eigen::VectorXf::Zero(n);
  do {
    lambda_prev = lambda;
    lambda = 0.0;
    num = 0.0;
    for (int i = 0; i < n; ++i) {
      float res = residuals(i);

      if (std::isfinite(res)) {
        num += 1.0;
        lambda +=
            res * res * ((DEFAULT_DOF + 1.0) / (DEFAULT_DOF + lambda_prev * res * res));
      }
    }
    lambda = 1.0 / (lambda / num);
  } while (std::abs(lambda - lambda_prev) > 1e-3);

  for (int i = 0; i < n; i++) {
    float res = residuals(i);
    if (!std::isfinite(res)) continue;

    weights(i) = ((DEFAULT_DOF + 1.0) / (DEFAULT_DOF + lambda * res * res));
  }
}

void DirectOdometry::calcGradient(const cv::Mat &img, cv::Mat &gradient,
                                  int dir) {
  static const int off[2][2] = {{1, 0}, {0, 1}};

  int w = img.cols;
  int h = img.rows;
  float *input_ptr = (float *)img.data;
  gradient = cv::Mat::zeros(h, w, CV_32FC1);
  float *output_ptr = (float *)gradient.data;

  for (int y = off[1][dir]; y < h - off[1][dir]; y++) {
    for (int x = off[0][dir]; x < w - off[0][dir]; x++) {
      float v0 = input_ptr[(y - off[dir][1]) * w + (x - off[dir][0])];
      float v1 = input_ptr[(y + off[dir][1]) * w + (x + off[dir][0])];
      output_ptr[y * w + x] = (v1 - v0) / 2;
    }
  }

  return;
}

Eigen::MatrixXf DirectOdometry::calcJacobian(const Sophus::SE3f &xi,
                                             const int level) {

  Eigen::Vector4f klevel = this->intr_Pyramid[level];
  cv::Mat cImg_level = this->cImg_Pyramid[level];
  cv::Mat pDep_level = this->pDep_Pyramid[level];

  cv::Mat grad_x, grad_y;
  calcGradient(cImg_level, grad_x, 0);
  calcGradient(cImg_level, grad_y, 1);

  float *ptr_gradx = (float *)grad_x.data;
  float *ptr_grady = (float *)grad_y.data;
  float *ptr_pDep = (float *)pDep_level.data;

  // Camera intrinsics
  float fx = klevel(0);
  float fy = klevel(1);
  float cx = klevel(2);
  float cy = klevel(3);

  // Width and Height
  int w = cImg_level.cols;
  int h = cImg_level.rows;

  // RationMatrix and t
  Eigen::Matrix3f R = xi.rotationMatrix();
  Eigen::Vector3f t = xi.translation();

  // Jacobian
  Eigen::MatrixXf J(w * h, 6), JI(1, 2);
  Eigen::MatrixXf Jw(2, 6);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int pos = y * w + x;

      Eigen::Vector3f pt_3d((x + cx) / fx * ptr_pDep[pos],
                            (y + cy) / fy * ptr_pDep[pos], ptr_pDep[pos]);
      pt_3d = R * pt_3d + t;
      float X = pt_3d[0];
      float Y = pt_3d[1];
      float Z = pt_3d[2];

      Jw << fx * 1 / Z, 0, -fx * X / (Z * Z), -fx * (X * Y) / (Z * Z),
          fx * (1 + (X * X) / (Z * Z)), -fx * Y / Z,
          0, fy * 1 / Z, -fy * Y / (Z * Z), -fy * (1 + (Y * Y) / (Z * Z)),
          fy * X * Y / (Z * Z), fy * X / Z;

      if (Z > 0.0) {
        // project 3f point to 2d
        Eigen::Vector2f pt_2d(fx * X / Z - cx,
                              fy * X / Z - cy);
        JI(0, 0) = interpolate(ptr_gradx, pt_2d[0], pt_2d[1], w, h);
        JI(0, 1) = interpolate(ptr_grady, pt_2d[0], pt_2d[1], w, h);
      }

      J.row(pos) = JI * Jw;

      if (!std::isfinite(J.row(pos)[0]))
        J.row(pos).setZero();
    }
  }

  return J;
}

Sophus::SE3f DirectOdometry::optimize() {

  makePyramid();

  Sophus::SE3f xi(Eigen::Matrix4f::Identity());

  Eigen::Matrix<float, 6, 6> H;   // Hessian for GN optimization.
  Eigen::Matrix<float, 6, 1> inc; // step increments.

  for (int level = NUM_PYRAMID - 1; level >= 0; --level) {
    float error_prev = std::numeric_limits<float>::max();
    for (int itr = 0; itr < NUM_GNITERS; itr++) {
      // compute residuals and Jacobian
      Eigen::VectorXf residuals = calcResiduals(xi, level);

      Eigen::VectorXf weights;
      weighting(residuals, weights);
      residuals = residuals.cwiseProduct(weights);

      Eigen::MatrixXf J = calcJacobian(xi, level);
      // compute weighted Jacobian
      for (int i = 0; i < residuals.size(); ++i)
        for (int j = 0; j < J.cols(); ++j)
          J(i, j) = J(i, j) * weights[i];

      float error = residuals.transpose() * residuals;
      std::cout << level << ": " << error << std::endl;

      // compute update step.
      Eigen::VectorXf b = J.transpose() * residuals;
      H = J.transpose() * J;
      inc = H.ldlt().solve(b);

      // std::cout << inc << std::endl;

      xi = xi * Sophus::SE3f::exp(inc);

      // Break when convergence.
      if (error / error_prev > 0.995)
        break;

      error_prev = error;
    }
  }

  return xi;
}
