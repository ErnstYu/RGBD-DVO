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
    if (line[0] == '#' || line.size() <= 10)
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
  if (line[0] == '#' || line.size() <= 10)
    return false;

  std::stringstream ss(line);
  Vec3 t;
  float qx, qy, qz, qw;
  ss >> qw;

  ss >> t[0] >> t[1] >> t[2];
  ss >> qx >> qy >> qz >> qw;

  pose = Sophus::SE3f(Eigen::Quaternionf(qw, qx, qy, qz), t);

  return true;
}

bool loadGroundTruth(const std::string &dataset, Poses &poses) {
  poses.clear();

  std::ifstream fin((dataset + "/aligned_gt.txt").c_str());
  if (!fin.is_open())
    return false;

  Sophus::SE3f pose, T_0_w;
  std::string line;
  poses.push_back(Sophus::SE3f(Eigen::Matrix4f::Identity()));

  do {
    std::getline(fin, line);
  } while (!poseFromStr(line, T_0_w));

  while (!fin.eof()) {
    do {
      std::getline(fin, line);
    } while (!fin.eof() && !poseFromStr(line, pose));

    if (fin.eof())
      break;

    poses.push_back(T_0_w.inverse() * pose);
  }
  fin.close();
  return true;
}

void savePoses(const Poses &poses, const std::string &fn) {

  std::ofstream fout(fn.c_str());
  for (size_t i = 0; i < poses.size(); ++i) {
    fout << poses[i].translation().transpose() << "   ";
    Eigen::Quaternionf quat(poses[i].unit_quaternion());
    fout << quat.x() << "   " << quat.y() << "   " << quat.z() << "   "
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

template <typename Sim3Derived, typename SE3Derived>
Sophus::SE3<typename Eigen::ScalarBinaryOpTraits<
    typename Sim3Derived::Scalar, typename SE3Derived::Scalar>::ReturnType>
operator*(const Sophus::Sim3Base<Sim3Derived> &a,
          const Sophus::SE3Base<SE3Derived> &b) {
  return {a.quaternion().normalized() * b.unit_quaternion(),
          a.rxso3() * b.translation() + a.translation()};
}

void evaluate(const Poses &gt_poses, Poses &poses) {
  // 0. Centroids
  size_t N = gt_poses.size();
  Mat3X dataPt(3, N), gtPt(3, N);
  for (size_t i = 0; i < N; ++i) {
    dataPt.col(i) = poses[i].translation();
    gtPt.col(i) = gt_poses[i].translation();
  }
  const Vec3 dataMean = dataPt.rowwise().mean();
  const Vec3 gtMean = gtPt.rowwise().mean();

  // center both clouds to 0 centroid
  Mat3X dataCtr = dataPt.colwise() - dataMean;
  Mat3X gtCtr = gtPt.colwise() - gtMean;

  // 1. Rotation

  // sum of outer products of columns
  const Mat3 W = gtCtr * dataCtr.transpose();

  const auto svd = W.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  // last entry to ensure we don't get a reflection, only rotations
  const Mat3 S = Eigen::DiagonalMatrix<float, 3, 3>(
      1, 1,
      svd.matrixU().determinant() * svd.matrixV().determinant() < 0 ? -1 : 1);

  const Mat3 R = svd.matrixU() * S * svd.matrixV().transpose();

  const Mat3X data_rotated = R * dataCtr;

  // 2. Scale (regular, non-symmetric variant)

  // sum of column-wise dot products
  const double dots = (gtCtr.cwiseProduct(data_rotated)).sum();

  // sum of column-wise norms
  const double norms = dataCtr.colwise().squaredNorm().sum();

  // scale
  const double s = dots / norms;

  // 3. Translation
  const Vec3 t = gtMean - s * R * dataMean;

  // 4. Translational error
  const Mat3X diff = gtPt - ((s * R * dataPt).colwise() + t);
  const Eigen::ArrayXf errors = diff.colwise().norm().transpose();
  float rmse = std::sqrt(errors.square().sum() / errors.rows());
  std::cout << rmse << std::endl;

  Sophus::Sim3f xi = Sophus::Sim3f(Sophus::RxSO3f(s, R), t);
  for (size_t i = 0; i < N; ++i)
    poses[i] = xi * poses[i];
}
