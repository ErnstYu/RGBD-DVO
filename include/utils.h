#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <fstream>
#include <sophus/se3.hpp>
#include <sstream>
#include <vector>

bool loadFilePaths(const std::string dataset, const std::string filename,
                   std::vector<std::string> &paths);

bool getPose(const std::string line, Sophus::SE3f &pose);

bool loadGroundTruth(const std::string dataset, const std::string filename,
                     std::vector<Sophus::SE3f> &poses);

void savePoses(const std::vector<Sophus::SE3f> &tforms,
               std::vector<Sophus::SE3f> &poses);

float interpolate(const float *img_ptr, float x, float y, int w, int h);

#endif // UTILS_H