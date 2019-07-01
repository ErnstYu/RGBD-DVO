#include <Eigen/Geometry>
#include <directOdometry.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <utils.h>

const std::string DATASET = "../data/fr1_desk";
const Eigen::Vector4f INTR(525.0, 525.0, 319.5, 239.5); // fx, fy, cx, cy
const float FACTOR = 5000.0;

std::vector<Sophus::SE3f> tforms, gt_tforms;

int main() {
  std::vector<std::string> inputRGBPaths, inputDepPaths;

  if (!loadFilePaths(DATASET, inputRGBPaths, inputDepPaths))
    exit(-1);

  size_t NUM_IMG = inputRGBPaths.size();
  cv::Mat pImg, pDep, cImg, cDep;

  pImg = cv::imread(inputRGBPaths[0], cv::IMREAD_GRAYSCALE);
  pDep = cv::imread(inputDepPaths[0], cv::IMREAD_ANYDEPTH);

  for (size_t i = 1; i < NUM_IMG - 1; ++i) {
    cImg = cv::imread(inputRGBPaths[i], cv::IMREAD_GRAYSCALE); // 8 bit rgb
    cDep = cv::imread(inputDepPaths[i], cv::IMREAD_ANYDEPTH);  // 16 bit depth

    DirectOdometry dvo(pImg, pDep, cImg, INTR, FACTOR);
    tforms.push_back(dvo.optimize());

    pImg = cImg.clone();
    pDep = cDep.clone();
    std::cout << i << std::endl;
  }

  std::vector<Sophus::SE3f> poses, gt_poses;
  absPoses(tforms, poses);
  savePoses(poses, "poses.txt");

  loadGroundTruth(DATASET, "groundtruth.txt", gt_poses);

  return 0;
}