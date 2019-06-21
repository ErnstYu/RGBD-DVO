#include <Eigen/Geometry>
#include <directOdometry.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sstream>

const std::string DATASET = "../data/test";
const Eigen::Vector4f INTR(525.0, 525.0, 319.5, 239.5); // fx, fy, cx, cy
const float FACTOR = 5000.0;

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

void savePoses(const std::vector<Sophus::SE3f> &tforms)
{
  Sophus::SE3f pose(Eigen::Matrix4f::Identity());
  std::ofstream fout("poses.txt");
  for (size_t i = 0; i < tforms.size(); ++i)
  {
    pose = tforms[i] * pose;
    fout << pose.translation().transpose() << '\t';
    Eigen::Quaternionf quat(pose.unit_quaternion());
    fout << quat.x() << '\t' << quat.y() << '\t' << quat.z() << '\t'
              << quat.w() << std::endl;
  }
  fout.close();
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

  size_t NUM_IMG = std::min(inputRGBPaths.size(), inputDepPaths.size());
  cv::Mat pImg, pDep, cImg, cDep;

  pImg = cv::imread(inputRGBPaths[0], cv::IMREAD_GRAYSCALE);
  pDep = cv::imread(inputDepPaths[0], cv::IMREAD_ANYDEPTH);

  for (size_t i = 1; i < NUM_IMG - 1; ++i) {
    cImg = cv::imread(inputRGBPaths[i], cv::IMREAD_GRAYSCALE);  // 8 bit rgb
    cDep = cv::imread(inputDepPaths[i], cv::IMREAD_ANYDEPTH);   // 16 bit depth

    DirectOdometry dvo(pImg, pDep, cImg, INTR, FACTOR);
    tforms.push_back(dvo.optimize());

    std::cout << tforms[i - 1].translation().transpose() << ' ';
    Eigen::Quaternionf quat(tforms[i - 1].unit_quaternion());
    std::cout << quat.x() << ' ' << quat.y() << ' ' << quat.z() << ' '
              << quat.w() << std::endl;

    pImg = cImg.clone();
    pDep = cDep.clone();
  }
  savePoses(tforms);
  return 0;
}