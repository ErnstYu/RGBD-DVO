#include <frame.h>

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

void Frame::calcGradient(const cv::Mat &img, cv::Mat &grad_x, cv::Mat &grad_y) {

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

void Frame::makePyramid(const cv::Mat &gray, const cv::Mat &depth,
                        const Intrinsics &intr) {
  intr_Pyramid.push_back(intr);
  gray_Pyramid.push_back(gray);
  depth_Pyramid.push_back(depth);

  cv::Mat grad_x, grad_y;
  calcGradient(gray, grad_x, grad_y);
  gradx_Pyramid.push_back(grad_x);
  grady_Pyramid.push_back(grad_y);

  for (int i = 1; i < NUM_PYRAMID; ++i) {
    // downsample camera matrix
    intr_Pyramid.push_back(intr_Pyramid[i - 1]);
    intr_Pyramid[i][2] -= 0.5;
    intr_Pyramid[i][3] -= 0.5;
    intr_Pyramid[i] *= 0.5;

    // downsample grayscale images
    cv::Mat grayDown = downsampleImg(gray_Pyramid[i - 1]);
    gray_Pyramid.push_back(grayDown);

    // downsample depth images
    cv::Mat depDown = downsampleDepth(depth_Pyramid[i - 1]);
    depth_Pyramid.push_back(depDown);

    // calculate image gradient
    calcGradient(grayDown, grad_x, grad_y);
    gradx_Pyramid.push_back(grad_x);
    grady_Pyramid.push_back(grad_y);
  }
}
