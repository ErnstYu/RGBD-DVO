#include <Eigen/Geometry>
#include <directOdometry.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sstream>

const std::string DATASET = "../data/test";
const Eigen::Vector4f INTR(525.0, 525.0, 319.5, 239.5); // fx, fy, cx, cy

std::vector<Sophus::SE3f> tforms;

bool loadFilePaths(const std::string dataset, const std::string filename,
                   std::vector<std::string> &paths) {
  std::string line;
  std::ifstream fin((dataset + '/' + filename).c_str());
  if (!fin.is_open())
    return false;

  while (!fin.eof()) {
    std::getline(fin, line);
    if (line[0] == '#')
      continue;

    std::stringstream ss(line);
    std::string buf;
    ss >> buf;
    ss >> buf;
    paths.push_back(dataset + '/' + buf);
  }
  fin.close();
  return true;
}

int main() {
  std::vector<std::string> inputRGBPaths, inputDepPaths;

  if (!loadFilePaths(DATASET, "rgb.txt", inputRGBPaths)) {
    std::cerr << "Cannot open " + DATASET + "/rgb.txt!\n";
    return -1;
  }
  if (!loadFilePaths(DATASET, "depth.txt", inputDepPaths)) {
    std::cerr << "Cannot open " + DATASET + "/depth.txt!\n";
    return -1;
  }

  size_t NUM_IMG = inputRGBPaths.size() < inputDepPaths.size()
                       ? inputRGBPaths.size()
                       : inputDepPaths.size();
  cv::Mat pImg, pDep, cImg, cDep;

  pImg = cv::imread(inputRGBPaths[0], cv::IMREAD_GRAYSCALE);
  pDep = cv::imread(inputDepPaths[0], cv::IMREAD_GRAYSCALE);

  for (size_t i = 1; i < NUM_IMG - 1; ++i) {

    cImg = cv::imread(inputRGBPaths[i], cv::IMREAD_GRAYSCALE);
    cDep = cv::imread(inputDepPaths[i], cv::IMREAD_GRAYSCALE);

    DirectOdometry dvo(pImg, pDep, cImg, INTR);
    tforms.push_back(dvo.optimize());

    pImg = cImg.clone();
    pDep = cDep.clone();
  }

  return 0;
}