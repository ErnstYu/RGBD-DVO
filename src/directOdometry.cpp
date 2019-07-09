#include <directOdometry.h>

float interpolate(const float *img_ptr, float x, float y, int w, int h) {
  float val = nan("1");
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
    val = total / sumWeights;

  return val;
}

cv::Mat downsampleImg(const cv::Mat &img) {

  int w = img.cols, h = img.rows;
  int w_ds = w / 2, h_ds = h / 2;

  float *input_ptr = (float *)img.ptr();
  cv::Mat img_ds = cv::Mat::zeros(h_ds, w_ds, img.type());
  float *output_ptr = (float *)img_ds.data;

  for (int y = 0; y < h_ds; ++y) {
    for (int x = 0; x < w_ds; ++x) {
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

  for (int y = 0; y < h_ds; ++y) {
    for (int x = 0; x < w_ds; ++x) {
      int top_left = 2 * y * w + 2 * x;
      int top_right = top_left + 1;
      int btm_left = (2 * y + 1) * w + 2 * x;
      int btm_right = btm_left + 1;
      float total = 0.0, count = 0.0;

      // To keep the border of 3D shape, a pixel without depth is ignored.
      if (input_ptr[top_left] != 0.0) {
        total += input_ptr[top_left];
        count += 1.0;
      }

      if (input_ptr[top_right] != 0.0) {
        total += input_ptr[top_right];
        count += 1.0;
      }

      if (input_ptr[btm_left] != 0.0) {
        total += input_ptr[btm_left];
        count += 1.0;
      }

      if (input_ptr[btm_right] != 0.0) {
        total += input_ptr[btm_right];
        count += 1.0;
      }

      if (count > 0) {
        output_ptr[y * w_ds + x] = total / count;
      }
    }
  }

  return depth_ds;
}

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

void DirectOdometry::calcGradient(const cv::Mat &img, cv::Mat &grad_x,
                                  cv::Mat &grad_y) {

  float *input_ptr = (float *)img.data;

  int w = img.cols;
  int h = img.rows;

  grad_x = cv::Mat::zeros(h, w, CV_32FC1);
  float *output_ptr = (float *)grad_x.data;
  for (int y = 0; y < h; ++y) {
    output_ptr[y * h] = input_ptr[y * w + 1] - input_ptr[y * w];
    output_ptr[y * w + w - 1] =
        input_ptr[y * w + w - 1] - input_ptr[y * w + w - 2];

    for (int x = 1; x < w - 1; ++x) {
      float v0 = input_ptr[y * w + (x - 1)];
      float v1 = input_ptr[y * w + (x + 1)];
      output_ptr[y * w + x] = (v1 - v0) / 2;
    }
  }

  grad_y = cv::Mat::zeros(h, w, CV_32FC1);
  output_ptr = (float *)grad_y.data;
  for (int x = 0; x < w; ++x) {
    output_ptr[x] = input_ptr[w + x] - input_ptr[x];
    output_ptr[(h - 1) * w + x] =
        input_ptr[(h - 1) * w + x] - input_ptr[(h - 2) * w + x];

    for (int y = 1; y < h - 1; ++y) {
      float v0 = input_ptr[(y - 1) * w + x];
      float v1 = input_ptr[(y + 1) * w + x];
      output_ptr[y * w + x] = (v1 - v0) / 2;
    }
  }
}

void DirectOdometry::makePyramid() {
  intr_Pyramid.push_back(this->intr);
  pImg_Pyramid.push_back(this->pImg);
  pDep_Pyramid.push_back(this->pDep);
  cImg_Pyramid.push_back(this->cImg);

  cv::Mat grad_x, grad_y;
  calcGradient(this->cImg, grad_x, grad_y);
  gradx_Pyramid.push_back(grad_x);
  grady_Pyramid.push_back(grad_y);

  for (int i = 1; i < NUM_PYRAMID; ++i) {
    // downsample camera matrix
    intr_Pyramid.push_back(intr_Pyramid[i - 1]);
    intr_Pyramid[i][2] += 0.5;
    intr_Pyramid[i][3] += 0.5;
    intr_Pyramid[i] *= 0.5;
    intr_Pyramid[i][2] -= 0.5;
    intr_Pyramid[i][3] -= 0.5;

    // downsample grayscale images
    cv::Mat pImgDown = downsampleImg(pImg_Pyramid[i - 1]);
    cv::Mat cImgDown = downsampleImg(cImg_Pyramid[i - 1]);
    pImg_Pyramid.push_back(pImgDown);
    cImg_Pyramid.push_back(cImgDown);

    // downsample depth images
    cv::Mat pDepDown = downsampleDepth(pDep_Pyramid[i - 1]);
    pDep_Pyramid.push_back(pDepDown);

    // calculate image gradient
    calcGradient(cImgDown, grad_x, grad_y);
    gradx_Pyramid.push_back(grad_x);
    grady_Pyramid.push_back(grad_y);
  }
}

void DirectOdometry::calcResiduals(const Sophus::SE3f &xi, const int level,
                                   Eigen::VectorXf &residuals) {

  Eigen::Vector4f intr_level = intr_Pyramid[level];
  cv::Mat cImg_level = cImg_Pyramid[level];
  cv::Mat pImg_level = pImg_Pyramid[level];
  cv::Mat pDep_level = pDep_Pyramid[level];

  int w = pImg_level.cols;
  int h = pImg_level.rows;

  // camera intrinsics
  float fx = intr_level(0);
  float fy = intr_level(1);
  float cx = intr_level(2);
  float cy = intr_level(3);

  Eigen::Matrix3f R = xi.rotationMatrix();
  Eigen::Vector3f t = xi.translation();

  float *ptr_pImg = (float *)pImg_level.data;
  float *ptr_pDep = (float *)pDep_level.data;
  float *ptr_cImg = (float *)cImg_level.data;

  residuals = Eigen::VectorXf::Ones(w * h) * nan("2");

  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u) {
      int pos = v * w + u;
      if (ptr_pDep[pos] < 1e-6)
        continue;

      Eigen::Vector3f pt3d(((float)u - cx) / fx * ptr_pDep[pos],
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

void DirectOdometry::calcFinalRes(const Sophus::SE3f &xi) {
  Eigen::VectorXf res;
  calcResiduals(xi, 0, res);

  float *ptr_res = (float *)finalResidual.data;

  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      int pos = v * W + u;
      if (!std::isnan(res[pos]))
        ptr_res[pos] = abs(res[pos] * 255.0);
    }
  }
}

void DirectOdometry::showError(const Sophus::SE3f &xi, const int level) {
  static int itr = 0;
  Eigen::VectorXf res;
  calcResiduals(xi, 0, res);

  cv::Mat err = cv::Mat::zeros(H, W, CV_32FC1);
  float *ptr_err = (float *)err.data;

  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      int pos = v * W + u;
      if (!std::isnan(res[pos]))
        ptr_err[pos] = abs(res[pos] * 255.0);
    }
  }
  cv::imwrite(std::to_string(itr) + '_' + std::to_string(level) + ".png", err);
  itr += 1;
}

void DirectOdometry::calcJacobian(const Sophus::SE3f &xi, const int level,
                                  Eigen::MatrixXf &J) {

  Eigen::Vector4f intr_level = intr_Pyramid[level];
  cv::Mat cImg_level = cImg_Pyramid[level];
  cv::Mat pDep_level = pDep_Pyramid[level];
  cv::Mat gradx_level = gradx_Pyramid[level];
  cv::Mat grady_level = grady_Pyramid[level];

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
  Eigen::Matrix3f R = xi.rotationMatrix();
  Eigen::Vector3f t = xi.translation();

  // Jacobian
  J = Eigen::MatrixXf::Zero(w * h, 6);
  Eigen::MatrixXf JI(1, 2), Jw(2, 6);

  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      int pos = v * w + u;
      if (ptr_pDep[pos] < 1e-6)
        continue;

      Eigen::Vector3f pt3d(((float)u - cx) / fx * ptr_pDep[pos],
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

Sophus::SE3f DirectOdometry::optimize() {

  makePyramid();

  Sophus::SE3f xi(Eigen::Matrix4f::Identity());
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
        xi = Sophus::SE3f::exp(xi_inc).inverse() * xi;
        break;
      }
      if (error / error_prev > 0.997)
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

      // Jt * (W * J) * xi_inc = - Jt * (W * r)
      xi_inc = (Jt * J).ldlt().solve(-Jt * residuals);

      xi = Sophus::SE3f::exp(xi_inc) * xi;
    }
  }
  calcFinalRes(xi);

  return xi;
}
