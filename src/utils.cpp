#include <utils.h>

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
    fout << pose.translation().transpose() << ' ';
    Eigen::Quaternionf quat(pose.unit_quaternion());
    fout << quat.x() << ' ' << quat.y() << ' ' << quat.z() << ' '
              << quat.w() << std::endl;
  }
  fout.close();
}

float interpolate(const float *img_ptr, float x, float y, int w, int h) {
  float val = nan("1");
  int x0 = floor(x), y0 = floor(y);
  int x1 = x0 + 1, y1 = y0 + 1;

  float x1_weight = x - x0, y1_weight = y - y0;
  float x0_weight = 1 - x1_weight, y0_weight = 1 - y1_weight;

  // Check if warped points are in the image.
  if (x0 < 0 or x0 >= w)
    x0_weight = 0;
  if (x1 < 0 or x1 >= w)
    x1_weight = 0;
  if (y0 < 0 or y0 >= h)
    y0_weight = 0;
  if (y1 < 0 or y1 >= h)
    y1_weight = 0;
  float w00 = x0_weight * y0_weight;
  float w10 = x1_weight * y0_weight;
  float w01 = x0_weight * y1_weight;
  float w11 = x1_weight * y1_weight;

  // Compute interpolated pixel intensity.
  float sumWeights = w00 + w10 + w01 + w11;
  float total = 0;
  if (w00 > 0)
    total += img_ptr[y0 * w + x0] * w00;
  if (w01 > 0)
    total += img_ptr[y1 * w + x0] * w01;
  if (w10 > 0)
    total += img_ptr[y0 * w + x1] * w10;
  if (w11 > 0)
    total += img_ptr[y1 * w + x1] * w11;

  if (sumWeights > 0)
    val = total / sumWeights;

  return val;
}
