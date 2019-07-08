#include <utils.h>

bool loadFilePaths(const std::string &dataset,
                   std::vector<std::string> &rgbPaths,
                   std::vector<std::string> &depPaths) {
  rgbPaths.clear();
  depPaths.clear();

  std::ifstream fin((dataset + "/assoc.txt").c_str());
  if (!fin.is_open()) {
    std::cerr << "Cannot open assoc.txt!\n";
    return false;
  }

  std::string line;
  while (!fin.eof()) {
    std::getline(fin, line);
    if (line.size() <= 10)
      continue;

    std::stringstream ss(line);
    std::string buf;
    ss >> buf;
    ss >> buf;
    rgbPaths.push_back(dataset + '/' + buf);
    ss >> buf;
    ss >> buf;
    depPaths.push_back(dataset + '/' + buf);
  }
  fin.close();
  return true;
}

bool poseFromStr(const std::string &line, Sophus::SE3f &pose) {
  if (line[0] == '#')
    return false;

  std::stringstream ss(line);
  Eigen::VectorXf t(3);
  float qx, qy, qz, qw;
  ss >> qw;

  ss >> t[0] >> t[1] >> t[2];
  ss >> qx >> qy >> qz >> qw;

  pose = Sophus::SE3f(Eigen::Quaternionf(qw, qx, qy, qz), t);

  return true;
}

bool loadGroundTruth(const std::string &dataset,
                     std::vector<Sophus::SE3f> &poses) {
  poses.clear();

  std::ifstream fin((dataset + "/groundtruth.txt").c_str());
  if (!fin.is_open())
    return false;

  Sophus::SE3f pose, first_pose;
  std::string line;

  do {
    std::getline(fin, line);
  } while (!poseFromStr(line, first_pose));
  poses.push_back(Sophus::SE3f(Eigen::Matrix4f::Identity()));

  while (!fin.eof()) {
    do {
      std::getline(fin, line);
    } while (!fin.eof() && !poseFromStr(line, pose));

    if (fin.eof())
      break;

    poses.push_back(first_pose.inverse() * pose);
  }
  fin.close();
  return true;
}

void savePoses(const std::vector<Sophus::SE3f> &poses, const std::string &fn) {

  std::ofstream fout(fn.c_str());
  for (size_t i = 0; i < poses.size(); ++i) {
    fout << poses[i].translation().transpose() << "   ";
    Eigen::Quaternionf quat(poses[i].unit_quaternion());
    fout << quat.x() << "   " << quat.y() << "   " << quat.z() << "   "
         << quat.w() << std::endl;
  }
  fout.close();
}

void renderCam(const Sophus::SE3f &xi, float lineWidth, const u_int8_t *color,
               float sizeFactor) {
  glPushMatrix();
  glMultMatrixd(xi.cast<double>().matrix().data());
  glColor3ubv(color);
  glLineWidth(lineWidth);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  const float sz = sizeFactor;
  const float width = 640, height = 480, fx = 500, fy = 500, cx = 320,
              cy = 240; // choose an arbitrary intrinsics because we don't need
                        // the camera be exactly same as the original one
  glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
  glEnd();
  glPopMatrix();
}