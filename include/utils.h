#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <fstream>
#include <sophus/se3.hpp>
#include <sstream>
#include <vector>

bool loadFilePaths(const std::string dataset,
                   std::vector<std::string> &rgbPaths,
                   std::vector<std::string> &depPaths);

bool poseFromStr(const std::string line, Sophus::SE3f &pose);

bool loadGroundTruth(const std::string dataset, const std::string filename,
                     std::vector<Sophus::SE3f> &poses);

void absPoses(const std::vector<Sophus::SE3f> &tforms,
              std::vector<Sophus::SE3f> &poses);

void savePoses(const std::vector<Sophus::SE3f> &poses, const std::string fn);

#endif // UTILS_H