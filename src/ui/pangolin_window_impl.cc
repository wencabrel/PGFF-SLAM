#include <pangolin/display/default_font.h>
#include <string>
#include <thread>
#include <glog/logging.h>

#include "common/options.h"
#include "common/std_types.h"
#include "core/lightning_math.hpp"
#include "ui/pangolin_window_impl.h"

namespace lightning::ui {

bool PangolinWindowImpl::Init() {
    // Check if DISPLAY is available
    const char* display = std::getenv("DISPLAY");
    if (!display || strlen(display) == 0) {
        LOG(WARNING) << "[UI] DISPLAY environment variable not set, UI disabled";
        ui_available_ = false;
        return false;
    }
    
    // Try to create a window with error handling
    try {
        pangolin::CreateWindowAndBind(win_name_, win_width_, win_height_);
    } catch (const std::exception& e) {
        LOG(WARNING) << "[UI] Failed to create window: " << e.what();
        LOG(WARNING) << "[UI] Running in headless mode (no visualization)";
        ui_available_ = false;
        return false;
    } catch (...) {
        LOG(WARNING) << "[UI] Unknown error creating window, running headless";
        ui_available_ = false;
        return false;
    }
    
    ui_available_ = true;

    // 3D mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // opengl buffer
    AllocateBuffer();

    // Initialize ImGui for modern UI controls
    imgui_panel_ = std::make_unique<ImGuiPanel>();
    if (!imgui_panel_->Init(win_width_, win_height_)) {
        LOG(WARNING) << "[UI] Failed to initialize ImGui panel";
        imgui_panel_.reset();
    }

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();

    // 雷达定位轨迹opengl设置 - Modern vibrant colors
    traj_newest_state_.reset(new ui::UiTrajectory(Vec3f(1.0, 0.4, 0.3)));   // Vibrant coral - Frontend
    traj_scans_.reset(new ui::UiTrajectory(Vec3f(0.0, 0.9, 0.6)));          // Neon cyan-green - Backend

    current_scan_.reset(new PointCloudType);  // 重置pcl点云指针
    current_scan_ui_.reset(new ui::UiCloud);  // 重置用于渲染的点云指针

    /// data log
    log_vel_.SetLabels(std::vector<std::string>{"vel_x", "vel_y", "vel_z"});
    log_vel_baselink_.SetLabels(std::vector<std::string>{"baselink_vel_x", "baselink_vel_y", "baselink_vel_z"});
    // PGFF metrics - more useful than generic confidence in SLAM mode
    log_confidence_.SetLabels(std::vector<std::string>{"PGFF Surprise"});
    log_error_.SetLabels(std::vector<std::string>{"Residual(cm)", "Uncertainty(sqrt)"});
    log_uncertainty_.SetLabels(std::vector<std::string>{"Pos_Uncertainty"});
    log_info_frontier_.SetLabels(std::vector<std::string>{"Info_Accuracy"});

    return true;
}

void PangolinWindowImpl::Reset(const std::vector<Keyframe::Ptr> &keyframes) {
    UL lock(mtx_reset_);
    cloud_map_ui_.clear();
    scans_.clear();
    current_scan_ui_ = nullptr;
    traj_scans_->Clear();

    for (const auto &keyframe : keyframes) {
        traj_scans_->AddPt(keyframe->GetOptPose());
    }

    std::size_t i = keyframes.size() > max_size_of_current_scan_ ? keyframes.size() - max_size_of_current_scan_ : 0;
    for (; i < keyframes.size(); ++i) {
        const auto &keyframe = keyframes.at(i);
        current_scan_ui_ = std::make_shared<ui::UiCloud>();
        CloudPtr tmp_cloud = std::make_shared<PointCloudType>(*(keyframe->GetCloud()));
        current_scan_ui_->SetCloud(math::VoxelGrid(tmp_cloud, 0.5), keyframe->GetOptPose());
        current_scan_ui_->SetRenderColor(ui::UiCloud::UseColor::HEIGHT_COLOR);

        scans_.emplace_back(current_scan_ui_);
    }

    newest_backend_pose_ = keyframes.back()->GetOptPose();
}

bool PangolinWindowImpl::DeInit() {
    // Cleanup ImGui if initialized
    if (imgui_panel_) {
        imgui_panel_->Shutdown();
        imgui_panel_.reset();
    }
    
    ReleaseBuffer();
    return true;
}

bool PangolinWindowImpl::UpdateGlobalMap() {
    if (!cloud_global_need_update_.load()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_map_cloud_);
    for (const auto &cp : cloud_global_map_) {
        if (cloud_map_ui_.find(cp.first) != cloud_map_ui_.end()) {
            continue;
        }

        std::shared_ptr<ui::UiCloud> ui_cloud(new ui::UiCloud);
        ui_cloud->SetCloud(cp.second, SE3());
        ui_cloud->SetRenderColor(ui::UiCloud::UseColor::GRAY_COLOR);
        cloud_map_ui_.emplace(cp.first, ui_cloud);
    }

    for (auto iter = cloud_map_ui_.begin(); iter != cloud_map_ui_.end();) {
        if (cloud_global_map_.find(iter->first) == cloud_global_map_.end()) {
            iter = cloud_map_ui_.erase(iter);
        } else {
            iter++;
        }
    }
    cloud_global_need_update_.store(false);

    return true;
}

bool PangolinWindowImpl::UpdateDynamicMap() {
    if (!cloud_dynamic_need_update_.load()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_map_cloud_);
    for (const auto &cp : cloud_dynamic_map_) {
        auto it = cloud_dyn_ui_.find(cp.first);
        if (it != cloud_dyn_ui_.end()) {
            // 存在也要更新
            it->second.reset(new ui::UiCloud);
            it->second->SetCloud(cp.second, SE3());
            // Use HEIGHT_COLOR for better depth perception and detail recognition
            it->second->SetRenderColor(ui::UiCloud::UseColor::HEIGHT_COLOR);
            continue;
        }

        /// 不存在则创建一个
        std::shared_ptr<ui::UiCloud> ui_cloud(new ui::UiCloud);
        ui_cloud->SetCloud(cp.second, SE3());
        // Use HEIGHT_COLOR for better depth perception and detail recognition
        ui_cloud->SetRenderColor(ui::UiCloud::UseColor::HEIGHT_COLOR);
        cloud_dyn_ui_.emplace(cp.first, ui_cloud);
    }

    for (auto iter = cloud_dyn_ui_.begin(); iter != cloud_dyn_ui_.end();) {
        if (cloud_dynamic_map_.find(iter->first) == cloud_dynamic_map_.end()) {
            iter = cloud_dyn_ui_.erase(iter);
        } else {
            iter++;
        };
    }

    cloud_dynamic_need_update_.store(false);
    return true;
}

bool PangolinWindowImpl::UpdateCurrentScan() {
    UL lock(mtx_current_scan_);
    if (current_scan_ != nullptr && !current_scan_->empty() && current_scan_need_update_) {
        if (current_scan_ui_) {
            current_scan_ui_->SetRenderColor(ui::UiCloud::UseColor::HEIGHT_COLOR);
            scans_.emplace_back(current_scan_ui_);
        }

        current_scan_ui_ = std::make_shared<ui::UiCloud>();
        current_scan_ui_->SetCloud(current_scan_, current_scan_pose_);
        current_scan_ui_->SetRenderColor(ui::UiCloud::UseColor::HEIGHT_COLOR);

        current_scan_need_update_.store(false);

        traj_scans_->AddPt(current_scan_pose_);

        newest_backend_pose_ = current_scan_pose_;
    }

    while (scans_.size() >= max_size_of_current_scan_) {
        scans_.pop_front();
    }

    return true;
}

void PangolinWindowImpl::UpdatePersistentMapLOD() {
    // Only update periodically to save CPU
    static int update_counter = 0;
    if (++update_counter % persistent_map_update_interval_ != 0) {
        return;
    }
    
    // Get list of new keyframes to process (thread-safe copy of IDs)
    std::vector<std::pair<size_t, CloudPtr>> new_keyframes;
    {
        UL lock(mtx_current_scan_);
        for (size_t i = last_processed_kf_id_; i < all_keyframes_.size(); ++i) {
            auto& kf = all_keyframes_[i];
            CloudPtr cloud = kf->GetCloud();  // This is already downsampled
            if (cloud && !cloud->empty()) {
                new_keyframes.push_back({i, cloud});
            }
        }
        last_processed_kf_id_ = all_keyframes_.size();
    }
    
    // Process new keyframes outside the lock
    for (const auto& [idx, cloud] : new_keyframes) {
        KeyframeLOD lod;
        lod.keyframe_id = idx;
        lod.valid = true;
        
        // Further downsample for LOD rendering (simple strided sampling)
        int stride = std::max(1, static_cast<int>(cloud->size() / 500));  // ~500 points per keyframe
        lod.local_points.reserve(cloud->size() / stride + 1);
        lod.colors.reserve(cloud->size() / stride + 1);
        
        for (size_t i = 0; i < cloud->size(); i += stride) {
            const auto& pt = cloud->points[i];
            lod.local_points.push_back(Vec3f(pt.x, pt.y, pt.z));
            
            // Height-based RGB coloring using turbo colormap style
            // Normalize height to [0, 1] range (assuming height range of -5 to 15 meters)
            float h = std::max(-5.0f, std::min(15.0f, pt.z));
            float t = (h + 5.0f) / 20.0f;  // Normalize to [0, 1]
            
            // Turbo colormap approximation: blue -> cyan -> green -> yellow -> red
            Vec4f color;
            if (t < 0.25f) {
                // Blue to cyan
                float s = t / 0.25f;
                color = Vec4f(0.18f + s * 0.02f, 0.18f + s * 0.62f, 0.85f - s * 0.15f, 0.85f);
            } else if (t < 0.5f) {
                // Cyan to green
                float s = (t - 0.25f) / 0.25f;
                color = Vec4f(0.2f + s * 0.3f, 0.8f + s * 0.1f, 0.7f - s * 0.5f, 0.85f);
            } else if (t < 0.75f) {
                // Green to yellow
                float s = (t - 0.5f) / 0.25f;
                color = Vec4f(0.5f + s * 0.5f, 0.9f, 0.2f - s * 0.15f, 0.85f);
            } else {
                // Yellow to red
                float s = (t - 0.75f) / 0.25f;
                color = Vec4f(1.0f, 0.9f - s * 0.7f, 0.05f, 0.85f);
            }
            lod.colors.push_back(color);
        }
        
        // Store in persistent map
        {
            std::lock_guard<std::mutex> plock(mtx_persistent_map_);
            if (idx >= persistent_map_lod_.size()) {
                persistent_map_lod_.resize(idx + 1);
            }
            persistent_map_lod_[idx] = std::move(lod);
        }
    }
}

void PangolinWindowImpl::RenderPersistentMap() {
    // Make a copy of keyframes for thread-safe access to poses
    std::vector<std::pair<size_t, SE3>> keyframe_poses;
    {
        UL lock(mtx_current_scan_);
        keyframe_poses.reserve(all_keyframes_.size());
        for (size_t i = 0; i < all_keyframes_.size(); ++i) {
            keyframe_poses.push_back({i, all_keyframes_[i]->GetOptPose()});
        }
    }
    
    // Now render with the persistent map lock
    std::lock_guard<std::mutex> plock(mtx_persistent_map_);
    
    glPointSize(1.5f);
    glBegin(GL_POINTS);
    
    for (const auto& [kf_idx, pose] : keyframe_poses) {
        if (kf_idx >= persistent_map_lod_.size()) continue;
        const auto& lod = persistent_map_lod_[kf_idx];
        if (!lod.valid || lod.local_points.empty()) continue;
        
        // Transform each point by the CURRENT optimized pose
        // This is the key: we use the latest pose, not a cached one
        Eigen::Matrix3f R = pose.rotationMatrix().cast<float>();
        Eigen::Vector3f t = pose.translation().cast<float>();
        
        for (size_t i = 0; i < lod.local_points.size(); ++i) {
            // Transform point from local to world frame using current pose
            Vec3f world_pt = R * lod.local_points[i] + t;
            const Vec4f& color = lod.colors[i];
            
            glColor4f(color[0], color[1], color[2], color[3]);
            glVertex3f(world_pt[0], world_pt[1], world_pt[2]);
        }
    }
    
    glEnd();
    glPointSize(1.0f);
}

bool PangolinWindowImpl::UpdateState() {
    if (!kf_result_need_update_.load()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_nav_state_);
    Vec3d pos = pose_.translation().eval();
    Vec3d vel_baselink = pose_.so3().inverse() * vel_;
    double roll = pose_.angleX();
    double pitch = pose_.angleY();
    double yaw = pose_.angleZ();

    // Debug: Print velocity values every 100 frames
    static int vel_debug_counter = 0;
    if (vel_debug_counter++ % 100 == 0) {
        LOG(INFO) << "[UI-VEL] Global vel: [" << vel_(0) << ", " << vel_(1) << ", " << vel_(2) 
                  << "] Baselink vel: [" << vel_baselink(0) << ", " << vel_baselink(1) << ", " << vel_baselink(2) << "]";
    }

    // 滤波器状态作曲线图
    log_vel_.Log(vel_(0), vel_(1), vel_(2));
    log_vel_baselink_.Log(vel_baselink(0), vel_baselink(1), vel_baselink(2));
    // Log PGFF metrics - surprise indicates novelty, residual indicates registration quality
    log_confidence_.Log(pgff_surprise_ * 100.0);  // Scale to percentage for better visibility
    
    // opt_residual_ is raw point-to-plane error in meters (typically 0.02-0.04m = 2-4cm)
    // Scale by 100 to convert to centimeters, so 0.03m -> 3.0 on chart
    double scaled_residual = opt_residual_ * 100.0;
    
    // Scale uncertainty: raw values range from 0.01m to 400m+
    // Use sqrt scale for better visibility: sqrt(uncertainty) puts 1m->1, 4m->2, 9m->3
    double scaled_uncertainty = std::sqrt(std::max(0.0, map_uncertainty_));
    scaled_uncertainty = std::min(4.0, scaled_uncertainty);  // Clamp to 0-4
    
    log_error_.Log(scaled_residual, scaled_uncertainty);  // Both on 0-4 scale now

    newest_frontend_pose_ = pose_;
    traj_newest_state_->AddPt(newest_frontend_pose_);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> kf_lock(mtx_current_scan_);
        keyframe_count_ = static_cast<int>(all_keyframes_.size());
        
        // Calculate total distance
        if (all_keyframes_.size() > 1) {
            double dist = 0.0;
            for (size_t i = 1; i < all_keyframes_.size(); ++i) {
                auto p1 = all_keyframes_[i-1]->GetOptPose().translation();
                auto p2 = all_keyframes_[i]->GetOptPose().translation();
                dist += (p2 - p1).norm();
            }
            total_distance_ = dist;
        }
        
        // Count loop closures from loop_info_
        loop_closure_count_ = static_cast<int>(loop_info_.size());
    }

    kf_result_need_update_.store(false);
    return false;
}

void PangolinWindowImpl::DrawAll() {
    // Render persistent map FIRST (background layer with all explored areas)
    // This uses current optimized poses, so it stays consistent after loop closure
    RenderPersistentMap();
    
    /// 地图
    for (const auto &pc : cloud_map_ui_) {
        pc.second->Render();
    }

    /// 动态地图
    for (const auto &pc : cloud_dyn_ui_) {
        pc.second->Render();
    }

    /// 缓存的scans (recent high-detail scans on top)
    for (const auto &s : scans_) {
        if (s) s->Render();
    }

    // Only render current scan if it exists and has points
    if (current_scan_ui_ && current_scan_ui_->HasPoints()) {
        current_scan_ui_->Render();
    }

    if (draw_frontend_traj_) {
        traj_newest_state_->Render();
        // Only render car if pose has been set (not at origin)
        if (newest_frontend_pose_.translation().norm() > 0.1) {
            frontend_car_.SetPose(newest_frontend_pose_);
            frontend_car_.Render();
        }
    }

    if (draw_backend_traj_) {
        traj_scans_->Render();
        // Only render car if pose has been set (not at origin)
        if (newest_backend_pose_.translation().norm() > 0.1) {
            backend_car_.SetPose(newest_backend_pose_);
            backend_car_.Render();
        }
    }

    // 关键帧
    {
        UL lock(mtx_current_scan_);

        if (all_keyframes_.size() > 1) {

            /// 闭环后的轨迹 - Purple/Magenta for optimized trajectory
            glLineWidth(4.0);
            glBegin(GL_LINE_STRIP);
            glColor3f(0.6, 0.2, 1.0);  // Electric purple

            for (int i = 0; i < all_keyframes_.size() - 1; ++i) {
                auto p1 = all_keyframes_[i]->GetOptPose().translation();
                auto p2 = all_keyframes_[i + 1]->GetOptPose().translation();

                glVertex3f(p1[0], p1[1], p1[2]);
                glVertex3f(p2[0], p2[1], p2[2]);
            }

            glEnd();
        }
    }

    // 文字
    RenderLabels();
    
    // Draw loop closures as connecting lines
    DrawLoopClosures();
}

void PangolinWindowImpl::DrawLoopClosures() {
    UL lock(mtx_current_scan_);
    
    if (loop_info_.empty() || all_keyframes_.empty()) {
        return;
    }
    
    // Draw loop closure connections as bright cyan lines
    glLineWidth(3.0);
    glBegin(GL_LINES);
    glColor4f(0.0, 0.95, 0.95, 0.8);  // Bright cyan with some transparency
    
    for (const auto& loop : loop_info_) {
        int kf1_id = loop.first;
        int kf2_id = loop.second;
        
        // Find the keyframes
        if (kf1_id < static_cast<int>(all_keyframes_.size()) && 
            kf2_id < static_cast<int>(all_keyframes_.size())) {
            auto p1 = all_keyframes_[kf1_id]->GetOptPose().translation();
            auto p2 = all_keyframes_[kf2_id]->GetOptPose().translation();
            
            // Draw line connecting the two loop closure keyframes
            glVertex3f(p1[0], p1[1], p1[2]);
            glVertex3f(p2[0], p2[1], p2[2]);
        }
    }
    
    glEnd();
    
    // Draw small spheres at loop closure locations
    glPointSize(8.0);
    glBegin(GL_POINTS);
    glColor4f(1.0, 0.84, 0.0, 1.0);  // Gold color for loop closure points
    
    for (const auto& loop : loop_info_) {
        int kf1_id = loop.first;
        int kf2_id = loop.second;
        
        if (kf1_id < static_cast<int>(all_keyframes_.size())) {
            auto p1 = all_keyframes_[kf1_id]->GetOptPose().translation();
            glVertex3f(p1[0], p1[1], p1[2]);
        }
        if (kf2_id < static_cast<int>(all_keyframes_.size())) {
            auto p2 = all_keyframes_[kf2_id]->GetOptPose().translation();
            glVertex3f(p2[0], p2[1], p2[2]);
        }
    }
    
    glEnd();
    glPointSize(1.0);  // Reset
}

void PangolinWindowImpl::RenderClouds() {
    UL lock(mtx_reset_);

    // 更新各种推送过来的状态
    UpdateGlobalMap();
    UpdateDynamicMap();
    UpdateState();
    UpdateCurrentScan();
    UpdatePersistentMapLOD();  // Update persistent map with new keyframes

    // 绘制
    pangolin::Display(dis_3d_main_name_).Activate(s_cam_main_);
    DrawAll();
}

void PangolinWindowImpl::RenderLabels() {
    // 定位状态标识，显示在3D窗口中
    auto &d_cam3d_main = pangolin::Display(dis_3d_main_name_);
    d_cam3d_main.Activate(s_cam_main_);
    const auto cur_width = d_cam3d_main.v.w;
    const auto cur_height = d_cam3d_main.v.h;

    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, cur_width, 0, cur_height, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glTranslatef(5, cur_height - 1.5 * gltext_label_global_.Height(), 1.0);
    glColor3ub(60, 80, 100);  // Subtle blue-gray grid lines
    gltext_label_global_.Draw();

    // Restore modelview / project matrices
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    
    // Render PGFF status in top-right corner
    RenderPGFFStatus();
    
    // Minimap disabled - uncomment to enable
    // RenderMinimap();
}

void PangolinWindowImpl::RenderPGFFStatus() {
    // Use the 3D view container for correct bounds
    auto &d_cam3d = pangolin::Display(dis_3d_name_);
    const auto cur_width = d_cam3d.v.w;
    const auto cur_height = d_cam3d.v.h;
    
    // Don't render if view is too small
    if (cur_width < 200 || cur_height < 100) return;
    
    // Activate the view
    d_cam3d.Activate();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, cur_width, 0, cur_height, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    auto &font = pangolin::default_font();
    
    // Status panel in top-right of the 3D view area (not overlapping with plots)
    char status_buf[512];
    snprintf(status_buf, sizeof(status_buf),
             "PGFF Status: %s\n"
             "Surprise: %.3f | Uncertainty: %.2fm\n"
             "KF: %d | Loops: %d | Dist: %.1fm",
             pgff_enabled_ ? "ACTIVE" : "OFF",
             pgff_surprise_,
             map_uncertainty_,
             keyframe_count_,
             loop_closure_count_,
             total_distance_);
    
    pangolin::GlText status_text = font.Text(status_buf);
    
    // Position in top-right corner with padding (keep away from right edge)
    float text_x = cur_width - status_text.Width() - 20;
    float text_y = cur_height - 20;
    
    // Clamp position to avoid going outside view
    if (text_x < 10) text_x = 10;
    
    // Draw semi-transparent background
    glColor4f(0.1, 0.1, 0.2, 0.7);
    glBegin(GL_QUADS);
    glVertex2f(text_x - 5, text_y - status_text.Height() - 5);
    glVertex2f(cur_width - 10, text_y - status_text.Height() - 5);
    glVertex2f(cur_width - 10, text_y + 10);
    glVertex2f(text_x - 5, text_y + 10);
    glEnd();
    
    // Draw text
    glTranslatef(text_x, text_y - status_text.Height() + 10, 1.0);
    if (pgff_enabled_) {
        glColor3f(0.0, 1.0, 0.5);  // Neon green when active
    } else {
        glColor3f(1.0, 0.2, 0.3);  // Hot pink-red when inactive
    }
    status_text.Draw();

    // Restore matrices
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void PangolinWindowImpl::RenderStatsPanel() {
    // Stats are now shown in menu panel, this is for additional overlay if needed
}

void PangolinWindowImpl::RenderMinimap() {
    // Top-down minimap in bottom-right corner of the 3D view area
    // Use the actual 3D view container (d_cam3d) which has the correct bounds
    auto &d_cam3d = pangolin::Display(dis_3d_name_);
    const auto cur_width = d_cam3d.v.w;
    const auto cur_height = d_cam3d.v.h;
    
    // Don't render if view is too small
    if (cur_width < 200 || cur_height < 200) return;
    
    // Minimap dimensions - position inside the 3D view area
    const float map_size = 150;
    const float map_x = cur_width - map_size - 30;  // More padding from right edge
    const float map_y = 20;
    
    // Activate the 3D view for correct coordinate system
    d_cam3d.Activate();
    
    // Save current state
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, cur_width, 0, cur_height, -1, 1);
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    // Draw minimap background
    glColor4f(0.1, 0.15, 0.25, 0.8);
    glBegin(GL_QUADS);
    glVertex2f(map_x, map_y);
    glVertex2f(map_x + map_size, map_y);
    glVertex2f(map_x + map_size, map_y + map_size);
    glVertex2f(map_x, map_y + map_size);
    glEnd();
    
    // Draw border
    glColor3f(0.0, 0.8, 0.9);  // Cyan trajectory on minimap
    glLineWidth(2.0);
    glBegin(GL_LINE_LOOP);
    glVertex2f(map_x, map_y);
    glVertex2f(map_x + map_size, map_y);
    glVertex2f(map_x + map_size, map_y + map_size);
    glVertex2f(map_x, map_y + map_size);
    glEnd();
    
    // Draw trajectory on minimap
    {
        UL lock(mtx_current_scan_);
        if (all_keyframes_.size() > 1) {
            // Calculate bounds
            double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
            for (const auto& kf : all_keyframes_) {
                auto p = kf->GetOptPose().translation();
                min_x = std::min(min_x, p[0]);
                max_x = std::max(max_x, p[0]);
                min_y = std::min(min_y, p[1]);
                max_y = std::max(max_y, p[1]);
            }
            
            double range_x = std::max(max_x - min_x, 1.0);
            double range_y = std::max(max_y - min_y, 1.0);
            double scale = (map_size - 20) / std::max(range_x, range_y);
            
            // Draw trajectory line
            glLineWidth(1.5);
            glBegin(GL_LINE_STRIP);
            glColor3f(0.8, 0.3, 1.0);  // Bright purple keyframes
            for (const auto& kf : all_keyframes_) {
                auto p = kf->GetOptPose().translation();
                float px = map_x + 10 + (p[0] - min_x) * scale;
                float py = map_y + 10 + (p[1] - min_y) * scale;
                glVertex2f(px, py);
            }
            glEnd();
            
            // Draw current position dot
            if (!all_keyframes_.empty()) {
                auto p = all_keyframes_.back()->GetOptPose().translation();
                float px = map_x + 10 + (p[0] - min_x) * scale;
                float py = map_y + 10 + (p[1] - min_y) * scale;
                
                glPointSize(6.0);
                glBegin(GL_POINTS);
                glColor3f(0.2, 1.0, 0.4);  // Lime green loop closures
                glVertex2f(px, py);
                glEnd();
                glPointSize(1.0);
            }
        }
    }
    
    // Draw "MINIMAP" label
    auto &font = pangolin::default_font();
    pangolin::GlText label = font.Text("MINIMAP");
    glColor3f(0.3, 0.5, 0.7);  // Steel blue border
    glTranslatef(map_x + 5, map_y + map_size - 15, 0);
    label.Draw();
    
    // Restore
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void PangolinWindowImpl::RenderProgressBar() {
    // Future: Progress bar for offline processing
}

void PangolinWindowImpl::CreateDisplayLayout() {
    // define camera render object (for view / scene browsing)
    // 定义视点的透视投影方式
    auto proj_mat_main = pangolin::ProjectionMatrix(win_width_, win_width_, cam_focus_, cam_focus_, win_width_ / 2,
                                                    win_width_ / 2, cam_z_near_, cam_z_far_);
    // 模型视图矩阵定义了视点的位置和朝向
    auto model_view_main = pangolin::ModelViewLookAt(0, 0, 100, 0, 0, 0, pangolin::AxisY);
    s_cam_main_ = pangolin::OpenGlRenderState(std::move(proj_mat_main), std::move(model_view_main));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam3d_main = pangolin::Display(dis_3d_main_name_)
                                       .SetBounds(0.0, 1.0, 0.0, 1.0)
                                       .SetHandler(new pangolin::Handler3D(s_cam_main_));

    pangolin::View &d_cam3d = pangolin::Display(dis_3d_name_)
                                  .SetBounds(0.0, 1.0, 0.0, 0.75)
                                  .SetLayout(pangolin::LayoutOverlay)
                                  .AddDisplay(d_cam3d_main);

    // OpenGL 'view' of data. We might have many views of the same data.
    // Velocity plot (Odom frame) - Enhanced with beautiful colors
    plotter_vel_ = std::make_unique<pangolin::Plotter>(&log_vel_, -10, 600, -11, 11, 75, 2);
    plotter_vel_->SetBounds(0.02, 0.98, 0.0, 1.0);
    plotter_vel_->Track("$i");
    plotter_vel_->SetBackgroundColour(pangolin::Colour(0.05f, 0.05f, 0.08f));  // Dark navy
    plotter_vel_->SetAxisColour(pangolin::Colour(0.3f, 0.4f, 0.6f));  // Steel blue axes
    plotter_vel_->SetTickColour(pangolin::Colour(0.4f, 0.5f, 0.7f));  // Lighter ticks
    
    // Velocity plot (Baselink frame) - Enhanced
    plotter_vel_baselink_ = std::make_unique<pangolin::Plotter>(&log_vel_baselink_, -10, 600, -11, 11, 75, 2);
    plotter_vel_baselink_->SetBounds(0.02, 0.98, 0.0, 1.0);
    plotter_vel_baselink_->Track("$i");
    plotter_vel_baselink_->SetBackgroundColour(pangolin::Colour(0.05f, 0.05f, 0.08f));
    plotter_vel_baselink_->SetAxisColour(pangolin::Colour(0.3f, 0.4f, 0.6f));
    plotter_vel_baselink_->SetTickColour(pangolin::Colour(0.4f, 0.5f, 0.7f));
    
    // Confidence/Surprise plot - Enhanced with warm colors
    plotter_confidence_ = std::make_unique<pangolin::Plotter>(&log_confidence_, -10, 600, 0, 5.0, 100, 0.5);
    plotter_confidence_->SetBounds(0.02, 0.98, 0.0, 1.0);
    plotter_confidence_->Track("$i");
    plotter_confidence_->SetBackgroundColour(pangolin::Colour(0.08f, 0.05f, 0.05f));  // Dark warm
    plotter_confidence_->SetAxisColour(pangolin::Colour(0.6f, 0.4f, 0.3f));  // Warm brown axes
    plotter_confidence_->SetTickColour(pangolin::Colour(0.7f, 0.5f, 0.4f));  // Lighter warm ticks
    
    // Error/Residual plot - Enhanced with cool colors
    plotter_err_ = std::make_unique<pangolin::Plotter>(&log_error_, -10, 600, 0, 4.0, 100, 0.5);
    plotter_err_->SetBounds(0.02, 0.98, 0.0, 1.0);
    plotter_err_->Track("$i");
    plotter_err_->SetBackgroundColour(pangolin::Colour(0.05f, 0.08f, 0.08f));  // Dark cyan
    plotter_err_->SetAxisColour(pangolin::Colour(0.3f, 0.6f, 0.6f));  // Cyan axes
    plotter_err_->SetTickColour(pangolin::Colour(0.4f, 0.7f, 0.7f));  // Lighter cyan ticks

    pangolin::View &d_plot = pangolin::Display(dis_plot_name_)
                                 .SetBounds(0.0, 1.0, 0.75, 1.0)
                                 .SetLayout(pangolin::LayoutEqualVertical)
                                 .AddDisplay(*plotter_confidence_)
                                 .AddDisplay(*plotter_err_)
                                 .AddDisplay(*plotter_vel_)
                                 .AddDisplay(*plotter_vel_baselink_);
    pangolin::Display(dis_main_name_)
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(menu_width_), 1.0)
        .AddDisplay(d_cam3d)
        .AddDisplay(d_plot);
}

void PangolinWindowImpl::Render() {
    pangolin::BindToContext(win_name_);

    // Issue specific OpenGl we might need
    // 启用OpenGL深度测试和混合功能，以支持透明度等效果。
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ImGui enabled flag
    bool use_imgui = (imgui_panel_ && imgui_panel_->IsInitialized());
    
    // Pangolin menu variables - declare outside to keep them in scope
    pangolin::Var<std::string> menu_header("menu.═══ LIGHTNING SLAM ═══", "");
    pangolin::Var<std::string> menu_view_section("menu.── View Controls ──", "");
    pangolin::Var<bool> menu_follow_loc("menu.Follow Robot", false, true);
    pangolin::Var<bool> menu_draw_frontend_traj("menu.Frontend Traj (Red)", true, true);
    pangolin::Var<bool> menu_draw_backend_traj("menu.Backend Traj (Green)", false, true);
    pangolin::Var<bool> menu_draw_loop_closures("menu.Loop Closures", true, true);
    pangolin::Var<std::string> menu_cam_section("menu.── Camera Presets ──", "");
    pangolin::Var<bool> menu_reset_3d_view("menu.Top-Down View", false, false);
    pangolin::Var<bool> menu_reset_front_view("menu.Side View", false, false);
    pangolin::Var<bool> menu_reset_follow_view("menu.Follow View", false, false);
    pangolin::Var<std::string> menu_pgff_section("menu.── PGFF Modules ──", "");
    pangolin::Var<bool> menu_pgff_enabled("menu.PGFF Enabled", true, true);
    pangolin::Var<bool> menu_uncertainty_map("menu.  Uncertainty Map", true, true);
    pangolin::Var<bool> menu_multi_hyp_lc("menu.  Multi-Hyp LC", true, true);
    pangolin::Var<bool> menu_info_frontier("menu.  Info Frontier", true, true);
    pangolin::Var<bool> menu_surprise_prior("menu.  Surprise Prior", true, true);
    pangolin::Var<std::string> menu_stats_section("menu.── Statistics ──", "");
    pangolin::Var<int> menu_kf_count("menu.Keyframes", 0);
    pangolin::Var<int> menu_loop_count("menu.Loop Closures", 0);
    pangolin::Var<double> menu_distance("menu.Distance (m)", 0.0);
    pangolin::Var<double> menu_fps("menu.FPS", 0.0);
    pangolin::Var<std::string> menu_metrics_section("menu.── PGFF Metrics ──", "");
    pangolin::Var<double> menu_surprise("menu.Surprise", 0.0);
    pangolin::Var<double> menu_uncertainty("menu.Uncertainty (m)", 0.0);
    pangolin::Var<double> menu_info_acc("menu.Info Accuracy (%)", 0.0);
    pangolin::Var<int> menu_active_hyp("menu.Active Hypotheses", 0);
    pangolin::Var<std::string> menu_play_section("menu.── Playback ──", "");
    pangolin::Var<bool> menu_step("menu.Step Mode", false, false);
    pangolin::Var<float> menu_play_speed("menu.Speed", 10.0, 0.1, 10.0);
    pangolin::Var<std::string> menu_display_section("menu.── Display ──", "");
    pangolin::Var<float> menu_intensity("menu.Point Opacity", 0.7, 0.1, 1.0);
    pangolin::Var<float> menu_point_size("menu.Point Size", 1.0, 0.5, 5.0);
    
    // Only create Pangolin panel if ImGui not available (fallback)
    if (!use_imgui) {
        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(menu_width_));
    }

    // display layout
    CreateDisplayLayout();

    exit_flag_.store(false);
    int frame_timeout_counter = 0;
    const int max_frame_timeout = 1000;  // Timeout after ~5 seconds of no activity
    
    while (!pangolin::ShouldQuit() && !exit_flag_) {
        // Frame timeout check to prevent indefinite blocking
        if (frame_timeout_counter++ > max_frame_timeout) {
            LOG(WARNING) << "[UI] Frame timeout detected, continuing...";
            frame_timeout_counter = 0;
        }
        
        // Clear entire screen - Deep space blue for professional SLAM visualization
        glClearColor(8.0 / 255.0, 12.0 / 255.0, 28.0 / 255.0, 1.0);  // Deep space blue
        // 清除了颜色缓冲区（GL_COLOR_BUFFER_BIT）和深度缓冲区（GL_DEPTH_BUFFER_BIT）。
        // 通常在每一帧渲染之前执行的操作，以准备渲染新的内容。
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Reset timeout counter on successful render
        frame_timeout_counter = 0;

        // ImGui or Pangolin menu controls
        if (use_imgui) {
            // Update ImGui state from internal variables
            auto& ui_state = imgui_panel_->GetState();
            ui_state.keyframe_count = keyframe_count_;
            ui_state.loop_count = loop_closure_count_;
            ui_state.distance_m = total_distance_;
            ui_state.fps = processing_fps_;
            ui_state.surprise = pgff_surprise_;
            ui_state.uncertainty_m = map_uncertainty_;
            ui_state.info_accuracy_pct = info_frontier_accuracy_ * 100.0;
            ui_state.active_hypotheses = active_hypotheses_;
            
            // Start ImGui frame (size will be updated after Pangolin layout)
            imgui_panel_->NewFrame();
            
            // Read controls from ImGui
            following_loc_ = ui_state.follow_robot;
            draw_frontend_traj_ = ui_state.show_frontend_traj;
            draw_backend_traj_ = ui_state.show_backend_traj;
            pgff_enabled_ = ui_state.pgff_enabled;
            uncertainty_mapping_enabled_ = ui_state.uncertainty_map;
            multi_hyp_lc_enabled_ = ui_state.multi_hyp_lc;
            info_frontier_enabled_ = ui_state.info_frontier;
            surprise_prior_enabled_ = ui_state.surprise_prior;
            
            // Camera preset buttons
            if (ui_state.btn_top_view) {
                s_cam_main_.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0, 1000, 0, 0, 0, pangolin::AxisY));
                ui_state.btn_top_view = false;
            }
            if (ui_state.btn_side_view) {
                s_cam_main_.SetModelViewMatrix(pangolin::ModelViewLookAt(-50, 0, 10, 50, 0, 10, pangolin::AxisZ));
                ui_state.btn_side_view = false;
            }
            if (ui_state.btn_follow_view) {
                following_loc_ = true;
                ui_state.follow_robot = true;
                ui_state.btn_follow_view = false;
            }
            
            // Playback controls
            debug::flg_next = ui_state.step_mode;
            debug::play_speed = ui_state.play_speed;
            ui::opacity = ui_state.point_opacity;
            
        } else {
            // OLD Pangolin menu control
        // menu control
        following_loc_ = menu_follow_loc;
        draw_frontend_traj_ = menu_draw_frontend_traj;
        draw_backend_traj_ = menu_draw_backend_traj;
        
        // PGFF module toggles
        pgff_enabled_ = menu_pgff_enabled;
        uncertainty_mapping_enabled_ = menu_uncertainty_map;
        multi_hyp_lc_enabled_ = menu_multi_hyp_lc;
        info_frontier_enabled_ = menu_info_frontier;
        surprise_prior_enabled_ = menu_surprise_prior;
        
        // Update statistics display
        menu_kf_count = keyframe_count_;
        menu_loop_count = loop_closure_count_;
        menu_distance = total_distance_;
        menu_fps = processing_fps_;
        
        // Update PGFF metrics display
        menu_surprise = pgff_surprise_;
        menu_uncertainty = map_uncertainty_;
        menu_info_acc = info_frontier_accuracy_ * 100.0;
        menu_active_hyp = active_hypotheses_;

        if (menu_reset_3d_view) {
            s_cam_main_.SetModelViewMatrix(pangolin::ModelViewLookAt(0, 0, 1000, 0, 0, 0, pangolin::AxisY));
            menu_reset_3d_view = false;
        }

        if (menu_reset_front_view) {
            s_cam_main_.SetModelViewMatrix(pangolin::ModelViewLookAt(-50, 0, 10, 50, 0, 10, pangolin::AxisZ));
            menu_reset_front_view = false;
        }
        
        if (menu_reset_follow_view) {
            following_loc_ = true;
            menu_follow_loc = true;
            menu_reset_follow_view = false;
        }

        if (menu_step) {
            debug::flg_next = true;
        } else {
            debug::flg_next = false;
        }

        debug::play_speed = menu_play_speed;
        ui::opacity = menu_intensity;
        } // End of Pangolin menu control (if !use_imgui)

        // Render pointcloud
        RenderClouds();
        
        // Render ImGui panel after Pangolin layout is set up
        if (use_imgui) {
            // Now query viewport after Pangolin has set up its displays
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);
            imgui_panel_->UpdateDisplaySize(viewport[2], viewport[3]);
            imgui_panel_->Render();
        }

        /// 处理相机跟随问题
        if (following_loc_) {
            Eigen::Vector3d translation = newest_frontend_pose_.translation();
            Sophus::SE3d newest_frontend_pose_new(Eigen::Quaterniond::Identity(),
                                                  Eigen::Vector3d(translation.x(), translation.y(), 0.0));
            s_cam_main_.Follow(newest_frontend_pose_new.matrix());
        }

        // Swap frames and Process Events
        // 完成当前帧的渲染并处理与窗口交互相关的事件

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // unset the current context from the main thread
    try {
        auto* window = pangolin::GetBoundWindow();
        if (window) {
            window->RemoveCurrent();
        }
        pangolin::DestroyWindow(GetWindowName());
    } catch (const std::exception& e) {
        LOG(WARNING) << "[UI] Error during cleanup: " << e.what();
    }
}

std::string PangolinWindowImpl::GetWindowName() const { return win_name_; }

void PangolinWindowImpl::AllocateBuffer() {
    std::string global_text(
        "Lightning SLAM - LiDAR-Inertial Odometry System\n"
        "Controls: Mouse drag to rotate, Scroll to zoom, Shift+drag to pan\n"
        "Red axis: Frontend pose | Green axis: Backend pose | Purple: Optimized trajectory");
    auto &font = pangolin::default_font();
    gltext_label_global_ = font.Text(global_text);
}

void PangolinWindowImpl::ReleaseBuffer() {}

}  // namespace lightning::ui
