#include <pcl/common/transforms.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "common/options.h"
#include "core/lightning_math.hpp"
#include "laser_mapping.h"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

namespace lightning {

bool LaserMapping::Init(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    ESKF::Options eskf_options;
    eskf_options.max_iterations_ = fasterlio::NUM_MAX_ITERATIONS;
    eskf_options.epsi_ = 1e-3 * Eigen::Matrix<double, 23, 1>::Ones();
    eskf_options.lidar_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs) { ObsModel(s, obs); };
    eskf_options.use_aa_ = use_aa_;
    kf_.Init(eskf_options);

    // Initialize PGFF - Predictive Geometric Flow Fields
    if (options_.enable_pgff_) {
        LOG(INFO) << "Initializing PGFF (Predictive Geometric Flow Fields)";
        pgff_ = std::make_unique<pgff::PredictiveLIO>(pgff::CreateDefaultPGFF());
        pgff_->SetEnabled(true);
        LOG(INFO) << "PGFF initialized - selective point processing enabled";
        
        // Initialize Information Frontier for predictive information (Option 3)
        pgff::InformationFrontier::Config info_config;
        info_config.cell_size = 5.0;           // 5m grid cells
        info_config.prediction_horizon = 10;   // Predict 10 cells ahead
        info_config.frontier_threshold = 0.7;  // High uncertainty = frontier
        info_frontier_ = std::make_unique<pgff::InformationFrontier>(info_config);
        LOG(INFO) << "Information Frontier initialized - predictive exploration enabled";
        
        // Initialize Learned Surprise Prior for adaptive thresholds (Option 4)
        pgff::LearnedSurprisePrior::Config prior_config;
        prior_config.region_size = 10.0;       // 10m spatial regions
        prior_config.learning_rate = 0.1;      // Moderate learning rate
        prior_config.novelty_bonus = 1.5;      // Boost novel regions
        prior_config.dynamic_penalty = 0.5;    // Penalize dynamic regions
        surprise_prior_ = std::make_unique<pgff::LearnedSurprisePrior>(prior_config);
        LOG(INFO) << "Learned Surprise Prior initialized - adaptive thresholds enabled";
    }

    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_scan;
    Vec3d lidar_T_wrt_IMU;
    Mat3d lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        fasterlio::NUM_MAX_ITERATIONS = yaml["fasterlio"]["max_iteration"].as<int>();
        fasterlio::ESTI_PLANE_THRESHOLD = yaml["fasterlio"]["esti_plane_threshold"].as<float>();

        filter_size_scan = yaml["fasterlio"]["filter_size_scan"].as<float>();
        filter_size_map_min_ = yaml["fasterlio"]["filter_size_map"].as<float>();
        keep_first_imu_estimation_ = yaml["fasterlio"]["keep_first_imu_estimation"].as<bool>();
        gyr_cov = yaml["fasterlio"]["gyr_cov"].as<float>();
        acc_cov = yaml["fasterlio"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["fasterlio"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["fasterlio"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["fasterlio"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["fasterlio"]["time_scale"].as<double>();
        lidar_type = yaml["fasterlio"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["fasterlio"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["fasterlio"]["point_filter_num"].as<int>();
        extrinsic_est_en_ = yaml["fasterlio"]["extrinsic_est_en"].as<bool>();
        extrinT_ = yaml["fasterlio"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["fasterlio"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["fasterlio"]["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["fasterlio"]["ivox_nearby_type"].as<int>();
        use_aa_ = yaml["fasterlio"]["use_aa"].as<bool>();

        skip_lidar_num_ = yaml["fasterlio"]["skip_lidar_num"].as<int>();
        enable_skip_lidar_ = skip_lidar_num_ > 0;
        
        // PGFF enable/disable from config
        if (yaml["fasterlio"]["enable_pgff"]) {
            options_.enable_pgff_ = yaml["fasterlio"]["enable_pgff"].as<bool>();
        }

    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_scan, filter_size_scan, filter_size_scan);

    lidar_T_wrt_IMU = math::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = math::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(Vec3d(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(Vec3d(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(Vec3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(Vec3d(b_acc_cov, b_acc_cov, b_acc_cov));

    return true;
}

LaserMapping::LaserMapping(Options options) : options_(options) {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
}

void LaserMapping::ProcessIMU(const lightning::IMUPtr &imu) {
    publish_count_++;

    double timestamp = imu->timestamp;

    UL lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    if (p_imu_->IsIMUInited()) {
        /// 更新最新imu状态
        kf_imu_.Predict(timestamp - last_timestamp_imu_, p_imu_->Q_, imu->angular_velocity, imu->linear_acceleration);

        // LOG(INFO) << "newest wrt lidar: " << timestamp - kf_.GetX().timestamp_;

        /// 更新ui
        if (ui_) {
            auto nav_state = kf_imu_.GetX();
            // Add PGFF metrics for real-time monitoring
            nav_state.pgff_surprise_ = current_frame_surprise_;
            nav_state.opt_residual_ = kf_.GetFinalRes();
            // Compute map uncertainty from ESKF covariance
            auto P = kf_.GetP();
            nav_state.map_uncertainty_ = std::sqrt(P(0,0) + P(1,1) + P(2,2));
            ui_->UpdateNavState(nav_state);
        }
    }

    last_timestamp_imu_ = timestamp;

    imu_buffer_.emplace_back(imu);
}

bool LaserMapping::Run() {
    if (!SyncPackages()) {
        return false;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return false;
    }

    /// the first scan
    if (flg_first_scan_) {
        LOG(INFO) << "first scan pts: " << scan_undistort_->size();

        state_point_ = kf_.GetX();
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            PointBodyToWorld(scan_undistort_->points[i], scan_down_world_->points[i]);
        }
        ivox_->AddPoints(scan_down_world_->points);

        first_lidar_time_ = measures_.lidar_end_time_;
        state_point_.timestamp_ = lidar_end_time_;
        flg_first_scan_ = false;
        return true;
    }

    if (enable_skip_lidar_) {
        skip_lidar_cnt_++;
        skip_lidar_cnt_ = skip_lidar_cnt_ % skip_lidar_num_;

        if (skip_lidar_cnt_ != 0) {
            /// 更新UI中的内容
            if (ui_) {
                auto nav_state = kf_.GetX();
                // Add PGFF metrics for real-time monitoring
                nav_state.pgff_surprise_ = current_frame_surprise_;
                nav_state.opt_residual_ = kf_.GetFinalRes();
                // Compute map uncertainty from ESKF covariance
                auto P = kf_.GetP();
                nav_state.map_uncertainty_ = std::sqrt(P(0,0) + P(1,1) + P(2,2));
                ui_->UpdateNavState(nav_state);
                ui_->UpdateScan(scan_undistort_, kf_.GetX().GetPose());
            }

            return false;
        }
    }

    // LOG(INFO) << "LIO get cloud at beg: " << std::setprecision(14) << measures_.lidar_begin_time_
    //           << ", end: " << measures_.lidar_end_time_;

    if (last_lidar_time_ > 0 && (measures_.lidar_begin_time_ - last_lidar_time_) > 0.5) {
        LOG(ERROR) << "检测到雷达断流，时长：" << (measures_.lidar_begin_time_ - last_lidar_time_);
    }

    last_lidar_time_ = measures_.lidar_begin_time_;

    flg_EKF_inited_ = (measures_.lidar_begin_time_ - first_lidar_time_) >= fasterlio::INIT_TIME;

    /// downsample
    voxel_scan_.setInputCloud(scan_undistort_);
    voxel_scan_.filter(*scan_down_body_);

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return false;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);

    Timer::Evaluate(
        [&, this]() {
            // 成员变量预分配
            residuals_.resize(cur_pts, 0);
            point_selected_surf_.resize(cur_pts, true);
            plane_coef_.resize(cur_pts, Vec4f::Zero());

            auto old_state = kf_.GetX();

            kf_.Update(ESKF::ObsType::LIDAR, 1e-3);
            state_point_ = kf_.GetX();

            if (keep_first_imu_estimation_ && all_keyframes_.size() < 5 &&
                (old_state.rot_.inverse() * state_point_.rot_).log().norm() > 0.3 * M_PI / 180) {
                kf_.ChangeX(old_state);
                state_point_ = old_state;

                LOG(INFO) << "set state as prediction";
            }

            // LOG(INFO) << "old yaw: " << old_state.rot_.angleZ() << ", new: " << state_point_.rot_.angleZ();

            state_point_.timestamp_ = measures_.lidar_end_time_;
            euler_cur_ = state_point_.rot_;
            pos_lidar_ = state_point_.pos_ + state_point_.rot_ * state_point_.offset_t_lidar_;
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " down " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    /// keyframes
    if (last_kf_ == nullptr) {
        MakeKF();
    } else {
        SE3 last_pose = last_kf_->GetLIOPose();
        SE3 cur_pose = state_point_.GetPose();
        if ((last_pose.translation() - cur_pose.translation()).norm() > options_.kf_dis_th_ ||
            (last_pose.so3().inverse() * cur_pose.so3()).log().norm() > options_.kf_angle_th_) {
            MakeKF();
        } else if (!options_.is_in_slam_mode_ && (state_point_.timestamp_ - last_kf_->GetState().timestamp_) > 2.0) {
            MakeKF();
        }
    }

    /// 更新kf_for_imu
    kf_imu_ = kf_;
    if (!measures_.imu_.empty()) {
        double t = measures_.imu_.back()->timestamp;
        for (auto &imu : imu_buffer_) {
            double dt = imu->timestamp - t;
            kf_imu_.Predict(dt, p_imu_->Q_, imu->angular_velocity, imu->linear_acceleration);
            t = imu->timestamp;
        }
    }

    if (ui_) {
        ui_->UpdateScan(scan_undistort_, state_point_.GetPose());
    }

    // Increment frame counter for PGFF and statistics
    frame_num_++;

    return true;
}

void LaserMapping::MakeKF() {
    Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_undistort_, state_point_);

    if (last_kf_) {
        // LOG(INFO) << "last kf lio: " << last_kf_->GetLIOPose().translation().transpose()
        //           << ", opt: " << last_kf_->GetOptPose().translation().transpose();

        /// opt pose 用之前的递推
        SE3 delta = last_kf_->GetLIOPose().inverse() * kf->GetLIOPose();
        kf->SetOptPose(last_kf_->GetOptPose() * delta);
    } else {
        kf->SetOptPose(kf->GetLIOPose());
    }

    kf->SetState(state_point_);
    
    // Set PGFF frame surprise for loop closing
    kf->SetFrameSurprise(current_frame_surprise_);
    
    // Compute pose uncertainty from ESKF covariance
    // Use trace of position covariance (first 3x3 block) as uncertainty metric
    auto P = kf_.GetP();
    double pos_uncertainty = std::sqrt(P(0,0) + P(1,1) + P(2,2));  // sqrt of trace
    kf->SetPoseUncertainty(pos_uncertainty);
    
    // Update Information Frontier with current observation (Option 3)
    UpdateInformationFrontier();
    
    // Update Learned Surprise Prior with current frame (Option 4)
    UpdateSurprisePrior();

    LOG(INFO) << "LIO: create kf " << kf->GetID() << ", state: " << state_point_.pos_.transpose()
              << ", kf opt pose: " << kf->GetOptPose().translation().transpose()
              << ", lio pose: " << kf->GetLIOPose().translation().transpose() << ", time: " << std::setprecision(14)
              << state_point_.timestamp_ << ", surprise: " << current_frame_surprise_;

    if (options_.is_in_slam_mode_) {
        all_keyframes_.emplace_back(kf);
    }

    last_kf_ = kf;
}

void LaserMapping::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = ToSec(msg->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            LOG(INFO) << "get cloud at " << std::setprecision(14) << timestamp
                      << ", latest imu: " << last_timestamp_imu_;

            CloudPtr cloud(new PointCloudType());
            preprocess_->Process(msg, cloud);

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

void LaserMapping::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = ToSec(msg->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            // LOG(INFO) << "get cloud at " << std::setprecision(14) << timestamp
            //           << ", latest imu: " << last_timestamp_imu_;

            CloudPtr cloud(new PointCloudType());
            preprocess_->Process(msg, cloud);

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

void LaserMapping::ProcessPointCloud2(CloudPtr cloud) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;

            double timestamp = math::ToSec(cloud->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.scan_ = lidar_buffer_.front();
        measures_.lidar_begin_time_ = time_buffer_.front();

        if (measures_.scan_->points.size() <= 1) {
            LOG(WARNING) << "Too few input point cloud!";
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else if (measures_.scan_->points.back().time / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_begin_time_ + measures_.scan_->points.back().time / double(1000);
            lidar_mean_scantime_ +=
                (measures_.scan_->points.back().time / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        lo::lidar_time_interval = lidar_mean_scantime_;

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = imu_buffer_.front()->timestamp;
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->timestamp;
        if (imu_time > lidar_end_time_) {
            break;
        }

        measures_.imu_.push_back(imu_buffer_.front());

        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    // LOG(INFO) << "sync: " << std::setprecision(14) << measures_.lidar_begin_time_ << ", " <<
    // measures_.lidar_end_time_;

    return true;
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    size_t cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(scan_down_body_->points[i], scan_down_world_->points[i]);

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = math::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= fasterlio::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < fasterlio::NUM_MATCH_POINTS; readd_i++) {
                    if (math::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }

            if (need_add) {
                points_to_add.emplace_back(point_world);  // FIXME 这并发可能有点问题
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * 
 * PGFF Enhancement: Uses Predictive Geometric Flow Fields to WEIGHT points
 * based on their surprise scores. All points get proper NN search and residual
 * computation - PGFF only affects the weighting in the robust kernel.
 * 
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(NavState &s, ESKF::CustomObservationModel &obs) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // PGFF: Compute point weights based on surprise scores
    // Higher surprise = higher weight (more informative points)
    if (options_.enable_pgff_ && pgff_ && pgff_->IsEnabled()) {
        ComputePGFFWeights(cnt_pts);
    } else {
        // Default: all points have equal weight
        pgff_point_weights_.assign(cnt_pts, 1.0f);
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot_ * s.offset_R_lidar_).cast<float>();
            auto t_wl = (s.rot_ * s.offset_t_lidar_ + s.pos_).cast<float>();

            // PGFF: ALL points get proper residual computation
            // The weighting happens later in the Jacobian/residual step
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                Vec3f p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                points_near.clear();

                /** Find the closest surfaces in the map **/
                ivox_->GetClosestPoint(point_world, points_near, fasterlio::NUM_MATCH_POINTS);
                point_selected_surf_[i] = points_near.size() >= fasterlio::MIN_NUM_MATCH_POINTS;
                if (point_selected_surf_[i]) {
                    point_selected_surf_[i] =
                        math::esti_plane(plane_coef_[i], points_near, fasterlio::ESTI_PLANE_THRESHOLD);
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        point_selected_surf_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    // PGFF: Log statistics if enabled
    int pgff_high_weight_count = 0;
    if (options_.enable_pgff_ && pgff_ && pgff_->IsEnabled()) {
        for (int i = 0; i < cnt_pts; i++) {
            if (point_selected_surf_[i] && pgff_point_weights_[i] > 1.2f) {
                pgff_high_weight_count++;
            }
        }
    }

    effect_feat_num_ = 0;

    // Store point weights for Jacobian computation
    std::vector<float> effect_weights;
    effect_weights.reserve(cnt_pts);
    
    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];
            
            // Store PGFF weight for this effective point
            effect_weights.push_back(pgff_point_weights_[i]);

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);
    effect_weights.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        obs.valid_ = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            obs.h_x_ = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            obs.residual_.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const Mat3f off_R = s.offset_R_lidar_.matrix().cast<float>();
            const Vec3f off_t = s.offset_t_lidar_.cast<float>();
            const Mat3f Rt = s.rot_.matrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Vec3f point_this_be = corr_pts_[i].head<3>();
                Mat3f point_be_crossmat = math::SKEW_SYM_MATRIX(point_this_be);
                Vec3f point_this = off_R * point_this_be + off_t;
                Mat3f point_crossmat = math::SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                Vec3f norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                Vec3f C(Rt * norm_vec);
                Vec3f A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    Vec3f B(point_be_crossmat * off_R.transpose() * C);
                    obs.h_x_.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0], B[1],
                        B[2], C[0], C[1], C[2];
                } else {
                    obs.h_x_.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0;
                }

                /// Cauchy's robust kernel with PGFF weighting
                /// PGFF weights: surprising points get higher weight (more informative)
                float res = -corr_pts_[i][3];
                float rho, drho;

                const float delta = 2.0;
                const float dsqr = delta * delta;
                const float dsqr_inv = 1.0 / dsqr;

                if (res >= 0) {
                    rho = dsqr * std::log(1 + res * dsqr_inv);
                    drho = 1.0 / (1 + res * dsqr_inv);
                } else {
                    rho = -dsqr * std::log(1 - res * dsqr_inv);
                    drho = 1.0 / (1 - res * dsqr_inv);
                }

                // PGFF: Apply weight to both residual and Jacobian
                // This makes surprising points contribute more to the optimization
                float pgff_weight = effect_weights[i];
                
                obs.residual_(i) = rho * pgff_weight;
                obs.h_x_.block<1, 12>(i, 0) = obs.h_x_.block<1, 12>(i, 0).eval() * drho * pgff_weight;
            });
        },
        "    ObsModel (IEKF Build Jacobian)");

    // PGFF: Log frame statistics
    if (options_.enable_pgff_ && pgff_ && pgff_->IsEnabled() && frame_num_ % 50 == 0) {
        LOG(INFO) << "[PGFF] Frame " << frame_num_ << ": effect=" << effect_feat_num_ 
                  << " high_weight=" << pgff_high_weight_count;
    }

    /// 填入中位数平方误差
    std::vector<double> res_sq2;
    for (size_t i = 0; i < cnt_pts; ++i) {
        if (point_selected_surf_[i]) {
            double r = residuals_[i];
            res_sq2.emplace_back(r * r);
        }
    }

    std::sort(res_sq2.begin(), res_sq2.end());
    obs.lidar_residual_mean_ = res_sq2[res_sq2.size() / 2];
    obs.lidar_residual_max_ = res_sq2[res_sq2.size() - 1];

    // PGFF: Store residuals for next frame prediction
    if (pgff_ && pgff_->IsEnabled()) {
        pgff_predicted_residuals_.clear();
        pgff_predicted_residuals_.reserve(cnt_pts);
        for (size_t i = 0; i < cnt_pts; ++i) {
            if (point_selected_surf_[i]) {
                pgff_predicted_residuals_.push_back(residuals_[i]);
            } else {
                pgff_predicted_residuals_.push_back(0.0f);
            }
        }
        
        // Update PGFF with current state for next frame prediction
        pgff_->GetFlowField().SetLastState(s.pos_.cast<float>(), s.rot_.matrix().cast<float>());
    }
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////

CloudPtr LaserMapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType);

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);

    for (auto &kf : all_keyframes_) {
        CloudPtr cloud = kf->GetCloud();

        CloudPtr cloud_filter(new PointCloudType);

        if (use_voxel) {
            voxel.setInputCloud(cloud);
            voxel.filter(*cloud_filter);

        } else {
            cloud_filter = cloud;
        }

        CloudPtr cloud_trans(new PointCloudType);

        if (use_lio_pose) {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetLIOPose().matrix());
        } else {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetOptPose().matrix());
        }

        *global_map += *cloud_trans;

        LOG(INFO) << "kf " << kf->GetID() << ", pose: " << kf->GetOptPose().translation().transpose();
    }

    CloudPtr global_map_filtered(new PointCloudType);
    if (use_voxel) {
        voxel.setInputCloud(global_map);
        voxel.filter(*global_map_filtered);
    } else {
        global_map_filtered = global_map;
    }

    global_map_filtered->is_dense = false;
    global_map_filtered->height = 1;
    global_map_filtered->width = global_map_filtered->size();

    LOG(INFO) << "global map: " << global_map_filtered->size();

    return global_map_filtered;
}

void LaserMapping::SaveMap() {
    /// 保存地图
    auto global_map = GetGlobalMap(true);

    pcl::io::savePCDFileBinaryCompressed("./data/lio.pcd", *global_map);

    LOG(INFO) << "lio map is saved to ./data/lio.pcd";
}

CloudPtr LaserMapping::GetRecentCloud() {
    if (lidar_buffer_.empty()) {
        return nullptr;
    }

    return lidar_buffer_.front();
}

/**
 * Compute PGFF weights for points based on surprise scores
 * 
 * PGFF weighting strategy:
 * - Points with high residual difference from prediction = high surprise = high weight
 * - Points with similar residual to prediction = low surprise = normal weight
 * - All points are processed, but surprising points contribute more to optimization
 * 
 * This is the CORRECT approach: we never skip residual computation,
 * we just weight points differently based on their information content.
 */
void LaserMapping::ComputePGFFWeights(int num_points) {
    pgff_point_weights_.resize(num_points, 1.0f);
    
    // If no previous residuals, all points get equal weight
    if (pgff_predicted_residuals_.empty()) {
        current_frame_surprise_ = 0.0;
        return;
    }
    
    // Resize predicted residuals to match current scan
    if (pgff_predicted_residuals_.size() != num_points) {
        // First frame or size mismatch - use equal weights
        pgff_predicted_residuals_.resize(num_points, 0.0f);
        current_frame_surprise_ = 0.0;
        return;
    }
    
    // Compute surprise-based weights
    // Using a simple but effective strategy:
    // weight = 1.0 + surprise_factor * |actual_residual - predicted_residual|
    
    const float surprise_scale = 5.0f;   // How much surprise affects weight
    const float min_weight = 0.5f;        // Minimum weight (expected points still contribute)
    const float max_weight = 3.0f;        // Maximum weight (cap for outliers)
    
    // Track frame-level surprise for loop closing
    double total_surprise = 0.0;
    int surprise_count = 0;
    int high_weight_count = 0;
    
    for (int i = 0; i < num_points; i++) {
        // Compute surprise from previous frame's residuals
        float predicted = (i < pgff_predicted_residuals_.size()) ? pgff_predicted_residuals_[i] : 0.0f;
        
        // Raw surprise based on predicted residuals
        float raw_surprise = std::abs(predicted);
        
        // Apply learned surprise prior for adaptive weighting (Option 4)
        float weight = 1.0f;
        if (surprise_prior_ && i < scan_down_body_->size()) {
            const auto& pt = scan_down_body_->points[i];
            Eigen::Vector3d world_pt = state_point_.rot_ * 
                (state_point_.offset_R_lidar_ * pt.getVector3fMap().cast<double>() + 
                 state_point_.offset_t_lidar_) + state_point_.pos_;
            
            // Get adaptive weight from learned prior
            weight = static_cast<float>(surprise_prior_->GetAdaptiveWeight(
                raw_surprise, world_pt, pgff::GeometryType::kUnknown));
        } else {
            // Fallback to simple heuristic
            float surprise = raw_surprise * surprise_scale;
            weight = 1.0f + std::tanh(surprise);
        }
        
        // Clamp to valid range
        pgff_point_weights_[i] = std::clamp(weight, min_weight, max_weight);
        
        if (pgff_point_weights_[i] > 1.5f) {
            high_weight_count++;
        }
        
        // Accumulate frame surprise
        total_surprise += raw_surprise;
        surprise_count++;
    }
    
    // Compute average frame surprise
    if (surprise_count > 0) {
        current_frame_surprise_ = total_surprise / surprise_count;
    } else {
        current_frame_surprise_ = 0.0;
    }
    
    // Log PGFF stats periodically
    if (frame_num_ % 20 == 0 && num_points > 0) {
        LOG(INFO) << "[PGFF] Frame " << frame_num_ 
                  << ": effect=" << num_points
                  << " high_weight=" << high_weight_count
                  << " surprise=" << std::setprecision(4) << current_frame_surprise_;
    }
}

void LaserMapping::UpdateInformationFrontier() {
    if (!info_frontier_) return;
    
    // Get predicted surprise before updating (for validation)
    current_predicted_surprise_ = info_frontier_->GetPredictedSurprise(state_point_.pos_);
    
    // Update frontier with actual observation
    info_frontier_->Update(state_point_.pos_, kf_id_, current_frame_surprise_);
    
    // Get prediction statistics
    auto stats = info_frontier_->GetPredictionStats();
    
    // Log periodically (every 10 keyframes)
    if (kf_id_ % 10 == 0 && stats.num_predictions > 0) {
        LOG(INFO) << "[InfoFrontier] KF " << kf_id_ 
                  << " cells=" << info_frontier_->GetCellCount()
                  << " frontiers=" << info_frontier_->GetFrontierCount()
                  << " pred_accuracy=" << std::setprecision(3) << stats.accuracy * 100 << "%"
                  << " pred_err=" << stats.mean_error;
    }
    
    // Store map uncertainty for UI
    current_map_uncertainty_ = 1.0 - stats.accuracy;  // Higher error = higher uncertainty
}

void LaserMapping::UpdateSurprisePrior() {
    if (!surprise_prior_) return;
    
    // Batch update surprise prior with current scan data
    // For efficiency, we sample a subset of points
    
    const int max_samples = 500;  // Limit samples for efficiency
    int num_points = scan_down_body_->size();
    int step = std::max(1, num_points / max_samples);
    
    std::vector<double> surprises;
    std::vector<Eigen::Vector3d> positions;
    std::vector<pgff::GeometryType> types;
    
    surprises.reserve(max_samples);
    positions.reserve(max_samples);
    types.reserve(max_samples);
    
    for (int i = 0; i < num_points; i += step) {
        // Get point position in world frame
        const auto& pt = scan_down_body_->points[i];
        Eigen::Vector3d world_pt = state_point_.rot_ * 
            (state_point_.offset_R_lidar_ * pt.getVector3fMap().cast<double>() + 
             state_point_.offset_t_lidar_) + state_point_.pos_;
        
        // Get residual as surprise proxy
        double surprise = 0.0;
        if (i < residuals_.size()) {
            surprise = std::abs(residuals_[i]);
        }
        
        // Classify geometry from plane normal (simplified)
        pgff::GeometryType geo_type = pgff::GeometryType::kUnknown;
        if (i < plane_coef_.size()) {
            Vec4f coef = plane_coef_[i];
            Vec3f normal(coef[0], coef[1], coef[2]);
            
            // Simple classification based on normal direction
            float z_component = std::abs(normal.z());
            float xy_component = std::sqrt(normal.x()*normal.x() + normal.y()*normal.y());
            
            if (z_component > 0.8f) {
                geo_type = pgff::GeometryType::kPlanar;  // Floor/ceiling
            } else if (xy_component > 0.8f) {
                geo_type = pgff::GeometryType::kPlanar;  // Wall
            } else {
                geo_type = pgff::GeometryType::kCorner;  // Transition
            }
        }
        
        surprises.push_back(surprise);
        positions.push_back(world_pt);
        types.push_back(geo_type);
    }
    
    // Batch update the prior
    surprise_prior_->UpdateBatch(surprises, positions, types);
    
    // Log learning progress periodically
    surprise_prior_->LogLearningProgress(frame_num_);
}

}  // namespace lightning