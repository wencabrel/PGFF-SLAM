#include "ui/pangolin_window_impl.h"
#include <glog/logging.h>

namespace lightning::ui {

PangolinWindow::PangolinWindow() { impl_ = std::make_shared<PangolinWindowImpl>(); }
PangolinWindow::~PangolinWindow() { Quit(); }

bool PangolinWindow::Init() {
    impl_->cloud_global_need_update_.store(false);
    impl_->kf_result_need_update_.store(false);
    impl_->lidarloc_need_update_.store(false);
    impl_->current_scan_need_update_.store(false);

    bool inited = impl_->Init();
    // 创建渲染线程 - only if UI initialization succeeded
    if (inited && impl_->ui_available_.load()) {
        LOG(INFO) << "[UI] Starting render thread...";
        impl_->render_thread_ = std::thread([this]() { impl_->Render(); });
    } else {
        LOG(WARNING) << "[UI] Running without visualization (headless mode)";
    }
    return inited;
}

void PangolinWindow::Reset(const std::vector<Keyframe::Ptr>& keyframes) { 
    if (impl_->ui_available_.load()) {
        impl_->Reset(keyframes); 
    }
}

void PangolinWindow::Quit() {
    impl_->exit_flag_.store(true);
    if (impl_->render_thread_.joinable()) {
        impl_->render_thread_.join();
    }
    if (impl_->ui_available_.load()) {
        impl_->DeInit();
    }
}

void PangolinWindow::UpdatePointCloudGlobal(const std::map<int, CloudPtr>& cloud) {
    if (!impl_->ui_available_.load()) return;
    // Use try_lock to avoid blocking the main SLAM thread
    std::unique_lock<std::mutex> lock(impl_->mtx_map_cloud_, std::try_to_lock);
    if (!lock.owns_lock()) return;  // Skip update if lock not available
    impl_->cloud_global_map_ = cloud;
    impl_->cloud_global_need_update_.store(true);
}

void PangolinWindow::UpdatePointCloudDynamic(const std::map<int, CloudPtr>& cloud) {
    if (!impl_->ui_available_.load()) return;
    // Use try_lock to avoid blocking the main SLAM thread
    std::unique_lock<std::mutex> lock(impl_->mtx_map_cloud_, std::try_to_lock);
    if (!lock.owns_lock()) return;  // Skip update if lock not available
    impl_->cloud_dynamic_map_.clear();  // need deep copy

    for (auto& cp : cloud) {
        CloudPtr c(new PointCloudType());
        *c = *cp.second;
        impl_->cloud_dynamic_map_.emplace(cp.first, c);
    }

    for (auto iter = impl_->cloud_dynamic_map_.begin(); iter != impl_->cloud_dynamic_map_.end();) {
        if (cloud.find(iter->first) == cloud.end()) {
            iter = impl_->cloud_dynamic_map_.erase(iter);
        } else {
            iter++;
        }
    }

    impl_->cloud_dynamic_need_update_.store(true);
}

void PangolinWindow::UpdateNavState(const NavState& state) {
    if (!impl_->ui_available_.load()) return;
    // Use try_lock to avoid blocking the main SLAM thread
    std::unique_lock<std::mutex> lock(impl_->mtx_nav_state_, std::try_to_lock);
    if (!lock.owns_lock()) return;  // Skip update if lock not available

    impl_->pose_ = state.GetPose();
    impl_->vel_ = state.GetVel();
    impl_->bias_acc_ = state.Getba();
    impl_->bias_gyr_ = state.Getbg();
    impl_->confidence_ = state.confidence_;
    
    // PGFF metrics for real-time monitoring
    impl_->pgff_surprise_ = state.pgff_surprise_;
    impl_->opt_residual_ = state.opt_residual_;
    impl_->map_uncertainty_ = state.map_uncertainty_;
    impl_->info_frontier_accuracy_ = state.info_frontier_accuracy_;

    impl_->kf_result_need_update_.store(true);
}

void PangolinWindow::UpdateRecentPose(const SE3& pose) {
    if (!impl_->ui_available_.load()) return;
    // Use try_lock to avoid blocking the main SLAM thread
    std::unique_lock<std::mutex> lock(impl_->mtx_nav_state_, std::try_to_lock);
    if (!lock.owns_lock()) return;  // Skip update if lock not available
    impl_->newest_frontend_pose_ = pose;
}

void PangolinWindow::UpdateScan(CloudPtr cloud, const SE3& pose) {
    if (!impl_->ui_available_.load()) return;
    // Use try_lock to avoid blocking - try both locks
    std::unique_lock<std::mutex> lock1(impl_->mtx_current_scan_, std::try_to_lock);
    if (!lock1.owns_lock()) return;  // Skip update if lock not available
    std::unique_lock<std::mutex> lock2(impl_->mtx_nav_state_, std::try_to_lock);
    if (!lock2.owns_lock()) return;  // Skip update if lock not available

    *impl_->current_scan_ = *cloud;  // need deep copy
    impl_->current_scan_pose_ = pose;
    impl_->current_scan_need_update_.store(true);
}

void PangolinWindow::UpdateKF(std::shared_ptr<Keyframe> kf) {
    if (!impl_->ui_available_.load()) return;
    // Use try_lock to avoid blocking the main SLAM thread
    std::unique_lock<std::mutex> lock(impl_->mtx_current_scan_, std::try_to_lock);
    if (!lock.owns_lock()) return;  // Skip update if lock not available
    impl_->all_keyframes_.emplace_back(kf);
}

void PangolinWindow::UpdateLoopClosingStats(int active_hypotheses) {
    if (!impl_->ui_available_.load()) return;
    impl_->active_hypotheses_ = active_hypotheses;
}

void PangolinWindow::UpdateLoopClosure(int idx1, int idx2) {
    if (!impl_->ui_available_.load()) return;
    std::unique_lock<std::mutex> lock(impl_->mtx_loop_info_);
    impl_->loop_info_.emplace_back(idx1, idx2);
}

void PangolinWindow::SetCurrentScanSize(int current_scan_size) { impl_->max_size_of_current_scan_ = current_scan_size; }

void PangolinWindow::SetTImuLidar(const SE3& T_imu_lidar) { impl_->T_imu_lidar_ = T_imu_lidar; }

bool PangolinWindow::ShouldQuit() { 
    if (!impl_->ui_available_.load()) return false;
    return pangolin::ShouldQuit(); 
}

}  // namespace lightning::ui
