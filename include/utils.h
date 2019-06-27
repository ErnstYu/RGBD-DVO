#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <sstream>
#include <vector>
#include <sophus/se3.hpp>

bool loadFilePaths(const std::string dataset, const std::string filename,
                   std::vector<std::string> &paths);

void savePoses(const std::vector<Sophus::SE3f> &tforms);

float interpolate(const float *img_ptr, float x, float y, int w, int h);

#endif // UTILS_H