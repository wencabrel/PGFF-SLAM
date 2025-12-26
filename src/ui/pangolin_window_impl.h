#pragma once

#include <pangolin/pangolin.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/pcl_base.hpp>

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "common/keyframe.h"
#include "common/loop_candidate.h"

#include "ui/pangolin_window.h"
#include "ui/ui_car.h"
#include "ui/ui_cloud.h"
#include "ui/ui_trajectory.h"
#include "ui/imgui_panel.h"

namespace lightning::ui {

/**
 */
class PangolinWindowImpl {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PangolinWindowImpl() = default;
    ~PangolinWindowImpl() = default;

    PangolinWindowImpl(const PangolinWindowImpl &) = delete;
    PangolinWindowImpl &operator=(const PangolinWindowImpl &) = delete;
    PangolinWindowImpl(PangolinWindowImpl &&) = delete;
    PangolinWindowImpl &operator=(PangolinWindowImpl &&) = delete;

    /// 初始化，创建可用于渲染的点云和轨迹
    bool Init();

    void Reset(const std::vector<Keyframe::Ptr> &keyframes);

    /// 注销
    bool DeInit();

    /// 渲染所有信息
    void Render();

    /// 获取窗口名称
    std::string GetWindowName() const;

   public:
    /// 后台渲染线程
    std::thread render_thread_;

    /// 一些辅助的锁和原子变量
    std::mutex mtx_map_cloud_;
    std::mutex mtx_current_scan_;
    std::mutex mtx_nav_state_;
    std::mutex mtx_gps_pose_;
    std::mutex mtx_loop_info_;

    std::mutex mtx_reset_;

    std::atomic<bool> exit_flag_;
    std::atomic<bool> ui_available_{false};  // Flag to indicate if UI is available

    std::atomic<bool> cloud_global_need_update_;   // 全局点云是否需要更新
    std::atomic<bool> cloud_dynamic_need_update_;  // 动态点云是否需要更新
    std::atomic<bool> kf_result_need_update_;      // 卡尔曼滤波结果
    std::atomic<bool> current_scan_need_update_;   // 更新当前扫描
    std::atomic<bool> lidarloc_need_update_;       // 雷达位置？

    pcl::PointCloud<PointType>::Ptr current_scan_ = nullptr;  // 当前scan
    SE3 newest_frontend_pose_;                                // 最新pose
    SE3 newest_backend_pose_;                                 // 最新pose
    SE3 current_scan_pose_;                                   // 当前scan对应的pose or Twb/Twi
    std::deque<std::pair<int, int>> loop_info_;
    std::vector<LoopCandidate> new_loop_candidate_;

    // 地图点云
    std::map<int, CloudPtr> cloud_global_map_;
    std::map<int, CloudPtr> cloud_dynamic_map_;

    /// 滤波器状态
    Sophus::SE3d pose_;
    double confidence_;
    Vec3d vel_;
    Vec3d bias_acc_;
    Vec3d bias_gyr_;
    Vec3d grav_;
    
    // PGFF monitoring metrics
    double pgff_surprise_ = 0.0;   // Frame surprise score
    double opt_residual_ = 0.0;    // Optimization residual
    double map_uncertainty_ = 0.0; // Map uncertainty from ESKF covariance
    
    // Enhanced UI statistics
    int keyframe_count_ = 0;           // Total keyframes
    int loop_closure_count_ = 0;       // Successful loop closures
    double total_distance_ = 0.0;      // Total distance traveled (m)
    double processing_fps_ = 0.0;      // Processing frame rate
    int active_hypotheses_ = 0;        // Active loop hypotheses
    double info_frontier_accuracy_ = 0.0;  // Info frontier prediction accuracy
    
    // PGFF module states (for UI toggles)
    bool pgff_enabled_ = true;
    bool uncertainty_mapping_enabled_ = true;
    bool multi_hyp_lc_enabled_ = true;
    bool info_frontier_enabled_ = true;
    bool surprise_prior_enabled_ = true;

    Sophus::SE3d T_imu_lidar_;
    int max_size_of_current_scan_ = 100;  // Recent scans for high detail
    std::vector<std::shared_ptr<Keyframe>> all_keyframes_;
    
    // Persistent map: stores downsampled LOCAL frame points for each keyframe
    // At render time, we apply the current optimized pose via OpenGL transforms
    // This ensures the map updates correctly after loop closure optimization
    struct KeyframeLOD {
        std::vector<Vec3f> local_points;  // Points in lidar/local frame (NOT world frame)
        std::vector<Vec4f> colors;        // Per-point colors
        size_t keyframe_id;
        bool valid = false;
    };
    std::vector<KeyframeLOD> persistent_map_lod_;  // One per keyframe
    std::mutex mtx_persistent_map_;
    size_t last_processed_kf_id_ = 0;
    static constexpr float lod_voxel_size_ = 0.5f;  // Downsample for LOD
    static constexpr int persistent_map_update_interval_ = 3;  // Update every N frames

    //////////////////////////////// 以下和render相关 ///////////////////////////
   private:
    /// 创建OpenGL Buffers
    void AllocateBuffer();
    void ReleaseBuffer();

    void CreateDisplayLayout();

    void DrawAll();  // 作图：画定位窗口
    
    // Persistent map rendering - renders all keyframe clouds with current optimized poses
    void UpdatePersistentMapLOD();  // Downsample new keyframes and store local-frame points
    void RenderPersistentMap();     // Render using OpenGL transforms with current poses

    /// 渲染点云，调用各种Update函数
    void RenderClouds();
    bool UpdateGlobalMap();
    bool UpdateDynamicMap();
    bool UpdateState();
    bool UpdateCurrentScan();

    void RenderLabels();

   private:
    /// 窗口layout相关
    int win_width_ = 1920;
    int win_height_ = 1080;
    static constexpr float cam_focus_ = 5000;
    static constexpr float cam_z_near_ = 1.0;
    static constexpr float cam_z_far_ = 1e10;
    static constexpr int menu_width_ = 210;
    const std::string win_name_ = "UI";
    const std::string dis_main_name_ = "main";
    const std::string dis_3d_name_ = "Cam 3D";
    const std::string dis_3d_main_name_ = "Cam 3D Main";  // main
    const std::string dis_plot_name_ = "Plot";
    const std::string dis_imgs_name = "Images";

    bool following_loc_ = true;       // 相机是否追踪定位结果
    bool draw_frontend_traj_ = true;  // 可视化前端轨迹
    bool draw_backend_traj_ = true;   // 可视化后端轨迹

    // text
    pangolin::GlText gltext_label_global_;
    pangolin::GlText gltext_stats_;        // Statistics text
    pangolin::GlText gltext_pgff_status_;  // PGFF status text

    // ImGui panel for modern UI controls
    std::unique_ptr<ImGuiPanel> imgui_panel_;

    // camera
    pangolin::OpenGlRenderState s_cam_main_;
    pangolin::OpenGlRenderState s_cam_minimap_;  // Minimap camera

    /// cloud rendering
    ui::UiCar backend_car_{Vec3f(0.34, 0.82, 0.50)};            // Emerald green - Backend
    ui::UiCar frontend_car_{Vec3f(0.98, 0.36, 0.26)};           // Coral red - Frontend
    std::map<int, std::shared_ptr<ui::UiCloud>> cloud_map_ui_;  // 用来渲染的点云地图
    std::map<int, std::shared_ptr<ui::UiCloud>> cloud_dyn_ui_;  // 用来渲染的点云地图
    std::shared_ptr<ui::UiCloud> current_scan_ui_;              // current scan
    std::deque<std::shared_ptr<ui::UiCloud>> scans_;            // current scan 保留的队列
    std::deque<std::pair<int, int>> loop_info_ui_;

    // trajectory
    std::shared_ptr<ui::UiTrajectory> traj_scans_ = nullptr;         // 激光扫描的轨迹
    std::shared_ptr<ui::UiTrajectory> traj_newest_state_ = nullptr;  // 最新state的轨迹

    // 滤波器状态相关 Data logger object
    pangolin::DataLog log_vel_;           // odom frame下的速度
    pangolin::DataLog log_vel_baselink_;  // baselink frame下的速度
    pangolin::DataLog log_confidence_;    // confidence
    pangolin::DataLog log_error_;         // 误差

    std::unique_ptr<pangolin::Plotter> plotter_vel_ = nullptr;
    std::unique_ptr<pangolin::Plotter> plotter_vel_baselink_ = nullptr;
    std::unique_ptr<pangolin::Plotter> plotter_confidence_ = nullptr;
    std::unique_ptr<pangolin::Plotter> plotter_err_ = nullptr;
    std::unique_ptr<pangolin::Plotter> plotter_err_eval_ = nullptr;
    
    // Enhanced UI - additional plots
    pangolin::DataLog log_uncertainty_;     // Uncertainty over time
    pangolin::DataLog log_info_frontier_;   // Info frontier metrics
    std::unique_ptr<pangolin::Plotter> plotter_uncertainty_ = nullptr;
    
    // Rendering helpers
    void RenderStatsPanel();      // Render statistics overlay
    void RenderPGFFStatus();      // Render PGFF module status
    void RenderMinimap();         // Render minimap
    void RenderProgressBar();     // Render progress bar
    void DrawLoopClosures();      // Visualize loop closures
};

}  // namespace lightning::ui
