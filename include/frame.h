#ifndef FRAME_H
#define FRAME_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <utils.h>

class Frame {
private:
  void calcGradient(const cv::Mat &img, cv::Mat &grad_x, cv::Mat &grad_y);
  void makePyramid(const cv::Mat &gray, const cv::Mat &depth,
                   const Intrinsics &intr);

public:
  int H, W;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> gray_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> depth_Pyramid;

  std::vector<Intrinsics, Eigen::aligned_allocator<Intrinsics>> intr_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> gradx_Pyramid;
  std::vector<cv::Mat, Eigen::aligned_allocator<cv::Mat>> grady_Pyramid;

  Frame(){}

  Frame(const std::string &rgbPath, const std::string &depPath,
        const Intrinsics &intr) {
    cv::Mat gray, depth;
    gray = cv::imread(rgbPath, cv::IMREAD_GRAYSCALE); // 8 bit
    depth = cv::imread(depPath, cv::IMREAD_ANYDEPTH); // 16 bit

    gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);
    depth.convertTo(depth, CV_32FC1, 1.0 / 5000.0);

    H = gray.rows; W = gray.cols;
    makePyramid(gray, depth, intr);
  }
};

#endif // FRAME_H