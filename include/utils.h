#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <fstream>
#include <pangolin/gl/gl.h>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <sstream>
#include <vector>

// Some type definitions:

using Intrinsics = Eigen::Vector4f;
using Poses = std::vector<Sophus::SE3f>;
using Transform = Sophus::SE3f;
using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;
using Mat3X = Eigen::Matrix<float, 3, Eigen::Dynamic>;

const u_int8_t COLOR_GT[3]{0, 250, 0}; // green
const u_int8_t COLOR_VO[3]{0, 0, 250}; // blue
const int NUM_PYRAMID = 5;

bool loadFilePaths(const std::string &dataset,
                   std::vector<std::string> &rgbPaths,
                   std::vector<std::string> &depPaths);

bool poseFromStr(const std::string &line, Sophus::SE3f &pose);

bool loadGroundTruth(const std::string &dataset, Poses &poses);

void savePoses(const Poses &poses, const std::string &fn);

float interpolate(const float *img_ptr, float x, float y, int w, int h);

void renderCam(const Sophus::SE3f &xi, float lineWidth, const u_int8_t *color,
               float sizeFactor);

void evaluate(const Poses &gt_poses, Poses &poses);

#endif // UTILS_H