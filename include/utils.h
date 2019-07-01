#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <fstream>
#include <pangolin/gl/gl.h>
#include <sophus/se3.hpp>
#include <sstream>
#include <vector>

const u_int8_t COLOR_GT[3]{0, 250, 0};         // green
const u_int8_t COLOR_VO[3]{0, 0, 250};         // blue

bool loadFilePaths(const std::string &dataset,
                   std::vector<std::string> &rgbPaths,
                   std::vector<std::string> &depPaths);

bool poseFromStr(const std::string &line, Sophus::SE3f &pose);

bool loadGroundTruth(const std::string &dataset, const std::string &filename,
                     std::vector<Sophus::SE3f> &poses);

void savePoses(const std::vector<Sophus::SE3f> &poses, const std::string &fn);

void renderCam(const Sophus::SE3f &xi, float lineWidth, const u_int8_t *color,
               float sizeFactor);

#endif // UTILS_H