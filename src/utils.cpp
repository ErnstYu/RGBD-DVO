#include <utils.h>

bool loadFilePaths(const std::string dataset, const std::string filename,
                   std::vector<std::string> &paths) {
  std::ifstream fin((dataset + '/' + filename).c_str());
  if (!fin.is_open())
    return false;

  std::string line;
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

bool getPose(const std::string line, Sophus::SE3f &pose) {
  if (line[0] == '#')
    return false;

  std::stringstream ss(line);
  std::string buf;
  Eigen::VectorXf t(3);
  float qx, qy, qz, qw;
  ss >> buf;

  ss >> buf;
  t[0] = atof(buf.c_str());

  ss >> buf;
  t[1] = atof(buf.c_str());

  ss >> buf;
  t[2] = atof(buf.c_str());

  ss >> buf;
  qx = atof(buf.c_str());

  ss >> buf;
  qy = atof(buf.c_str());

  ss >> buf;
  qz = atof(buf.c_str());

  ss >> buf;
  qw = atof(buf.c_str());

  pose = Sophus::SE3f(Eigen::Quaternionf(qw, qx, qy, qz), t);

  return true;
}

bool loadGroundTruth(const std::string dataset, const std::string filename,
                     std::vector<Sophus::SE3f> &poses) {
  std::ifstream fin((dataset + '/' + filename).c_str());
  if (!fin.is_open())
    return false;

  Sophus::SE3f pose, first_pose;
  std::string line;

  do {
    std::getline(fin, line);
  } while (!getPose(line, first_pose));
  poses.push_back(Sophus::SE3f(Eigen::Matrix4f::Identity()));

  while (!fin.eof()) {
    do {
      std::getline(fin, line);
    } while (!fin.eof() && !getPose(line, pose));

    if (fin.eof())
      break;

    poses.push_back(first_pose.inverse() * pose);
  }
  fin.close();
  return true;
}

void savePoses(const std::vector<Sophus::SE3f> &tforms,
               std::vector<Sophus::SE3f> &poses) {
  Sophus::SE3f pose(Eigen::Matrix4f::Identity());
  poses.push_back(pose);

  std::ofstream fout("poses.txt");
  for (size_t i = 0; i < tforms.size(); ++i) {
    pose = tforms[i] * pose;
    poses.push_back(pose);
    fout << pose.translation().transpose() << ' ';
    Eigen::Quaternionf quat(pose.unit_quaternion());
    fout << quat.x() << ' ' << quat.y() << ' ' << quat.z() << ' ' << quat.w()
         << std::endl;
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
