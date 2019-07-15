#include <Eigen/Geometry>
#include <directOdometry.h>
#include <opencv2/opencv.hpp>
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
const Intrinsics INTR(525.0, 525.0, 319.5, 239.5); // fx, fy, cx, cy

std::vector<std::string> inputRGBPaths, inputDepPaths;
Poses gt_poses, poses;
Transform T_c_p;
Frame pre, cur;
cv::Mat stepRes = cv::Mat::zeros(cv::Size(640, 480), CV_16UC1);
bool evaluated = false;

size_t num_img, idx = 1;

bool nextFrame() {
  if (idx >= num_img) {
    if (!evaluated) {
      evaluate(gt_poses, poses);
      evaluated = true;
    }
    return false;
  }

  cur = Frame(inputRGBPaths[idx], inputDepPaths[idx], INTR);

  DirectOdometry dvo(pre, cur);
  T_c_p = dvo.optimize(T_c_p);
  Sophus::SE3f pose = poses.back() * T_c_p.inverse();
  poses.push_back(pose);
  dvo.finalResidual.convertTo(stepRes, CV_16UC1, 255.0);

  pre = cur;

  std::cout << idx << std::endl;
  idx += 1;
  return true;
}

int main(int argc, char **argv) {
  bool show_gui = true;
  std::string dataset = "test";

  CLI::App app{"Direct Visual Odometry"};
  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset,
                 "Dataset path. Default: " + dataset);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (!loadFilePaths("../data/" + dataset, inputRGBPaths, inputDepPaths)) {
    std::cerr << "Cannot load dataset!\n";
    exit(-1);
  }
  if (!loadGroundTruth("../data/" + dataset, gt_poses)) {
    std::cerr << "Cannot load ground truth!\n";
    exit(-1);
  }
  poses.push_back(Sophus::SE3f(Eigen::Matrix4f::Identity()));
  num_img = inputRGBPaths.size();
  pre = Frame(inputRGBPaths[0], inputDepPaths[0], INTR);
  std::cout << pre.gray_Pyramid.size() << std::endl;

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
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.1, 100),
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

      for (auto tmp : poses)
        renderCam(tmp, 1.0, COLOR_VO, 1e-2);
      for (size_t j = 0; j < num_img; ++j)
        renderCam(gt_poses[j], 1.0, COLOR_GT, 1e-2);
      pangolin::glDrawAxis(1);

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