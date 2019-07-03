#include <Eigen/Geometry>
#include <directOdometry.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <utils.h>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/tbb.h>
#include <thread>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI11/CLI11.hpp>

bool nextFrame();

// ui buttons
constexpr int UI_WIDTH = 200;
pangolin::Var<bool> continue_next("ui.continue_next", false, false, true);
pangolin::Var<std::function<void(void)>> next_frame_btn("ui.next_frame",
                                                        &nextFrame);

// intrinsics
const Eigen::Vector4f INTR(525.0, 525.0, 319.5, 239.5); // fx, fy, cx, cy
const float FACTOR = 5000.0;

std::vector<std::string> inputRGBPaths, inputDepPaths;
std::vector<Sophus::SE3f> gt_poses;
Sophus::SE3f pose;
cv::Mat stepRes = cv::Mat::zeros(cv::Size(640, 480), CV_16UC1);

size_t idx = 1;

bool nextFrame() {
  if (idx >= inputRGBPaths.size())
    return false;

  cv::Mat pImg, pDep, cImg, cDep;
  pImg = cv::imread(inputRGBPaths[idx - 1], cv::IMREAD_GRAYSCALE); // 8 bit
  pDep = cv::imread(inputDepPaths[idx - 1], cv::IMREAD_ANYDEPTH);  // 16 bit
  cImg = cv::imread(inputRGBPaths[idx], cv::IMREAD_GRAYSCALE);

  DirectOdometry dvo(pImg, pDep, cImg, INTR, FACTOR);
  Sophus::SE3f tform = dvo.optimize();
  pose = tform * pose;
  dvo.finalResidual.convertTo(stepRes, CV_16UC1, 255.0);

  std::cout << idx << std::endl;
  idx += 1;
  return true;
}

int main(int argc, char **argv) {
  bool show_gui = true;
  std::string dataset = "../data/fr1_desk";

  CLI::App app{"Direct Visual Odometry"};
  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset,
                 "Dataset path. Default: " + dataset);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (!loadGroundTruth(dataset, "groundtruth.txt", gt_poses))
    exit(-1);
  if (!loadFilePaths(dataset, inputRGBPaths, inputDepPaths))
    exit(-1);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Direct Visual Odometry", 1500, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View &main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View &img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < 2) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);
      img_view.push_back(iv);
      img_view_display.AddDisplay(*iv);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.01, 1000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View &display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);
    main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      img_view_display.Activate();
      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f); // light gray background

      // current rgb image
      pangolin::TypedImage img = pangolin::LoadImage(inputRGBPaths[idx - 1]);
      img_view[0]->SetImage(img);

      // current residual image
      pangolin::ManagedImage<unsigned short> res(stepRes.cols, stepRes.rows);
      memcpy((void *)res.begin(), (void *)stepRes.data,
             stepRes.total() * stepRes.elemSize());
      img_view[1]->SetImage(res);

      renderCam(pose, 2.0, COLOR_VO, 0.1);

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = nextFrame();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      pangolin::FinishFrame();
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (nextFrame()) {
      // nothing here
    }
  }

  return 0;
}