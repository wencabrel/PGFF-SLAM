//
// Created by xiang on 25-4-21.
//

#include "core/loop_closing/loop_closing.h"
#include "common/keyframe.h"
#include "common/loop_candidate.h"
#include "utils/pointcloud_utils.h"

#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>

#include "core/opti_algo/algo_select.h"
#include "core/robust_kernel/cauchy.h"
#include "core/types/edge_se3.h"
#include "core/types/edge_se3_height_prior.h"
#include "core/types/vertex_se3.h"
#include "io/yaml_io.h"

namespace lightning {

LoopClosing::~LoopClosing() {
    if (options_.online_mode_) {
        kf_thread_.Quit();
    }
}

void LoopClosing::Init(const std::string yaml_path) {
    /// setup miao
    miao::OptimizerConfig config(miao::AlgorithmType::LEVENBERG_MARQUARDT,
                                 miao::LinearSolverType::LINEAR_SOLVER_SPARSE_EIGEN, false);
    config.incremental_mode_ = true;
    optimizer_ = miao::SetupOptimizer<6, 3>(config);

    info_motion_.setIdentity();
    info_motion_.block<3, 3>(0, 0) =
        Mat3d::Identity() * 1.0 / (options_.motion_trans_noise_ * options_.motion_trans_noise_);
    info_motion_.block<3, 3>(3, 3) =
        Mat3d::Identity() * 1.0 / (options_.motion_rot_noise_ * options_.motion_rot_noise_);

    info_loops_.setIdentity();
    info_loops_.block<3, 3>(0, 0) = Mat3d::Identity() * 1.0 / (options_.loop_trans_noise_ * options_.loop_trans_noise_);
    info_loops_.block<3, 3>(3, 3) = Mat3d::Identity() * 1.0 / (options_.loop_rot_noise_ * options_.loop_rot_noise_);

    if (!yaml_path.empty()) {
        YAML_IO yaml(yaml_path);

        options_.loop_kf_gap_ = yaml.GetValue<int>("loop_closing", "loop_kf_gap");
        options_.min_id_interval_ = yaml.GetValue<int>("loop_closing", "min_id_interval");
        options_.closest_id_th_ = yaml.GetValue<int>("loop_closing", "closest_id_th");
        options_.max_range_ = yaml.GetValue<double>("loop_closing", "max_range");
        options_.ndt_score_th_ = yaml.GetValue<double>("loop_closing", "ndt_score_th");
        options_.with_height_ = yaml.GetValue<bool>("loop_closing", "with_height");
    }
    
    // Initialize Multi-Hypothesis Loop Closing Manager
    if (options_.use_multi_hypothesis_) {
        LoopHypothesisManager::Config hyp_config;
        hyp_config.max_hypotheses_ = options_.max_hypotheses_;
        hyp_config.commit_threshold_ = options_.commit_threshold_;
        hyp_config.min_validations_ = options_.min_validations_;
        hyp_config.max_hypothesis_age_ = options_.max_hypothesis_age_;
        hypothesis_manager_ = std::make_unique<LoopHypothesisManager>(hyp_config);
        LOG(INFO) << "[Multi-Hypothesis LC] Initialized with max_hyp=" << options_.max_hypotheses_
                  << " commit_th=" << options_.commit_threshold_ 
                  << " min_val=" << options_.min_validations_;
    }

    if (options_.online_mode_) {
        LOG(INFO) << "loop closing module is running in online mode";
        kf_thread_.SetProcFunc([this](Keyframe::Ptr kf) { HandleKF(kf); });
        kf_thread_.SetName("handle loop closure");
        kf_thread_.Start();
    }
}

void LoopClosing::AddKF(Keyframe::Ptr kf) {
    if (options_.online_mode_) {
        kf_thread_.AddMessage(kf);
    } else {
        HandleKF(kf);
    }
}

void LoopClosing::HandleKF(Keyframe::Ptr kf) {
    if (kf == last_kf_) {
        return;
    }

    cur_kf_ = kf;
    all_keyframes_.emplace_back(kf);

    // 检测回环候选
    DetectLoopCandidates();

    if (options_.verbose_) {
        LOG(INFO) << "lc: get kf " << cur_kf_->GetID() << " candi: " << candidates_.size();
    }

    // 计算回环位姿
    ComputeLoopCandidates();
    
    // Multi-Hypothesis Processing
    if (options_.use_multi_hypothesis_ && hypothesis_manager_) {
        ProcessHypotheses();
    }

    // 位姿图优化
    PoseOptimization();

    last_kf_ = kf;
}

bool LoopClosing::IsSurpriseBasedLoopTrigger() {
    if (!options_.use_pgff_surprise_) {
        return false;
    }
    
    double current_surprise = cur_kf_->GetFrameSurprise();
    
    // Update moving average with exponential smoothing
    const double alpha = 0.3;  // Smoothing factor
    if (surprise_moving_avg_ == 0.0) {
        surprise_moving_avg_ = current_surprise;
    } else {
        surprise_moving_avg_ = alpha * current_surprise + (1.0 - alpha) * surprise_moving_avg_;
    }
    
    // Detect significant surprise drop (indicating familiar geometry)
    // A drop means we're likely revisiting a known location
    double surprise_drop = last_frame_surprise_ - current_surprise;
    double relative_drop = (last_frame_surprise_ > 0.01) ? 
                           surprise_drop / last_frame_surprise_ : 0.0;
    
    last_frame_surprise_ = current_surprise;
    
    // Trigger early loop detection if:
    // 1. Surprise dropped significantly from previous frame
    // 2. Or current surprise is much lower than moving average (familiar area)
    bool trigger = false;
    
    if (relative_drop > options_.surprise_drop_threshold_) {
        trigger = true;
        if (options_.verbose_) {
            LOG(INFO) << "PGFF: Surprise drop detected! " << relative_drop * 100 
                      << "% drop - possible loop closure area";
        }
    }
    
    if (surprise_moving_avg_ > 0.01 && 
        current_surprise < surprise_moving_avg_ * (1.0 - options_.surprise_drop_threshold_)) {
        trigger = true;
        if (options_.verbose_) {
            LOG(INFO) << "PGFF: Low surprise detected! Current: " << current_surprise 
                      << " vs avg: " << surprise_moving_avg_ << " - familiar geometry";
        }
    }
    
    return trigger;
}

void LoopClosing::DetectLoopCandidates() {
    candidates_.clear();

    auto& kfs_mapping = all_keyframes_;
    Keyframe::Ptr check_first = nullptr;

    if (last_loop_kf_ == nullptr) {
        last_loop_kf_ = cur_kf_;
        return;
    }

    // Determine the effective loop gap based on PGFF surprise
    int effective_loop_gap = options_.loop_kf_gap_;
    
    if (IsSurpriseBasedLoopTrigger()) {
        // Use reduced gap when surprise indicates familiar area
        effective_loop_gap = options_.surprise_early_gap_;
        if (options_.verbose_) {
            LOG(INFO) << "PGFF: Using early loop gap: " << effective_loop_gap;
        }
    }

    if (last_loop_kf_ && (cur_kf_->GetID() - last_loop_kf_->GetID()) <= effective_loop_gap) {
        LOG(INFO) << "skip because last loop kf: " << last_loop_kf_->GetID() 
                  << " (gap: " << effective_loop_gap << ")";
        return;
    }

    for (auto kf : kfs_mapping) {
        if (check_first != nullptr && std::abs(static_cast<int>(kf->GetID() - check_first->GetID())) <= options_.min_id_interval_) {
            // 同条轨迹内，跳过一定的ID区间
            continue;
        }

        if (std::abs(static_cast<int>(kf->GetID() - cur_kf_->GetID())) < options_.closest_id_th_) {
            /// 在同一条轨迹中，如果间隔太近，就不考虑回环
            break;
        }

        Vec3d dt = kf->GetOptPose().translation() - cur_kf_->GetOptPose().translation();
        double t2d = dt.head<2>().norm();  // x-y distance
        double range_th = options_.max_range_;

        if (t2d < range_th) {
            LoopCandidate c(kf->GetID(), cur_kf_->GetID());
            c.Tij_ = kf->GetLIOPose().inverse() * cur_kf_->GetLIOPose();

            candidates_.emplace_back(c);
            check_first = kf;
        }
    }

    if (!candidates_.empty()) {
        last_loop_kf_ = cur_kf_;
    }

    if (options_.verbose_ && !candidates_.empty()) {
        LOG(INFO) << "lc candi: " << candidates_.size();
    }
}

void LoopClosing::ComputeLoopCandidates() {
    if (candidates_.empty()) {
        return;
    }

    // 执行计算
    std::for_each(candidates_.begin(), candidates_.end(), [this](LoopCandidate& c) { ComputeForCandidate(c); });
    // 保存成功的候选
    std::vector<LoopCandidate> succ_candidates;
    for (const auto& lc : candidates_) {
        LOG(INFO) << "candi " << lc.idx1_ << ", " << lc.idx2_ << " s: " << lc.ndt_score_;
        if (lc.ndt_score_ > options_.ndt_score_th_) {
            succ_candidates.emplace_back(lc);
        }
    }

    if (options_.verbose_) {
        LOG(INFO) << "success: " << succ_candidates.size() << "/" << candidates_.size();
    }

    candidates_.swap(succ_candidates);
}

void LoopClosing::ComputeForCandidate(lightning::LoopCandidate& c) {
    LOG(INFO) << "aligning " << c.idx1_ << " with " << c.idx2_;
    const int submap_idx_range = 40;
    auto kf1 = all_keyframes_.at(c.idx1_), kf2 = all_keyframes_.at(c.idx2_);

    auto build_submap = [this](int given_id, bool build_in_world) -> CloudPtr {
        CloudPtr submap(new PointCloudType);
        for (int idx = -submap_idx_range; idx < submap_idx_range; idx += 4) {
            int id = idx + given_id;
            if (id < 0 || id > all_keyframes_.size()) {
                continue;
            }

            auto kf = all_keyframes_[id];
            CloudPtr cloud = kf->GetCloud();

            // RemoveGround(cloud, 0.1);

            if (cloud->empty()) {
                continue;
            }

            // 转到世界系下
            SE3 Twb = kf->GetLIOPose();

            if (!build_in_world) {
                Twb = all_keyframes_.at(given_id)->GetLIOPose().inverse() * Twb;
            }

            CloudPtr cloud_trans(new PointCloudType);
            pcl::transformPointCloud(*cloud, *cloud_trans, Twb.matrix());

            *submap += *cloud_trans;
        }
        return submap;
    };

    auto submap_kf1 = build_submap(kf1->GetID(), true);

    CloudPtr submap_kf2 = kf2->GetCloud();

    if (submap_kf1->empty() || submap_kf2->empty()) {
        c.ndt_score_ = 0;
        return;
    }

    Mat4f Tw2 = kf2->GetLIOPose().matrix().cast<float>();

    /// 不同分辨率下的匹配
    CloudPtr output(new PointCloudType);
    std::vector<double> res{10.0, 5.0, 2.0, 1.0};

    CloudPtr rough_map1, rough_map2;

    for (auto& r : res) {
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.05);
        ndt.setStepSize(0.7);
        ndt.setMaximumIterations(40);

        ndt.setResolution(r);
        rough_map1 = VoxelGrid(submap_kf1, r * 0.1);
        rough_map2 = VoxelGrid(submap_kf2, r * 0.1);
        ndt.setInputTarget(rough_map1);
        ndt.setInputSource(rough_map2);

        ndt.align(*output, Tw2);
        Tw2 = ndt.getFinalTransformation();

        c.ndt_score_ = ndt.getTransformationProbability();
        
        // Compute ICP fitness for multi-hypothesis scoring
        // Fitness is avg squared distance of correspondences (lower is better)
        c.icp_fitness_ = ndt.getFitnessScore();
    }

    Mat4d T = Tw2.cast<double>();
    Quatd q(T.block<3, 3>(0, 0));
    q.normalize();
    Vec3d t = T.block<3, 1>(0, 3);

    c.Tij_ = kf1->GetLIOPose().inverse() * SE3(q, t);
    
    // Set initial geometric consistency based on transform reasonableness
    double trans_norm = c.Tij_.translation().norm();
    double rot_angle = Eigen::AngleAxisd(c.Tij_.rotationMatrix()).angle();
    // Good consistency if transform is reasonable (not too large)
    c.geometric_consistency_ = std::max(0.0, 1.0 - trans_norm / 10.0) * 
                               std::max(0.0, 1.0 - rot_angle / M_PI);
    
    // Initial confidence computation
    c.ComputeConfidence();

    // pcl::io::savePCDFileBinaryCompressed(
    //     "./data/lc_" + std::to_string(c.idx1_) + "_" + std::to_string(c.idx2_) + "_out.pcd", *output);
    // pcl::io::savePCDFileBinaryCompressed(
    //     "./data/lc_" + std::to_string(c.idx1_) + "_" + std::to_string(c.idx2_) + "_tgt.pcd", *rough_map1);
}

void LoopClosing::ProcessHypotheses() {
    if (!hypothesis_manager_) return;
    
    int current_frame = cur_kf_->GetID();
    
    // Validate existing hypotheses with new candidates
    hypothesis_manager_->ValidateWithFrame(current_frame, cur_kf_->GetOptPose(), candidates_);
    
    // Add new high-quality candidates as hypotheses
    for (const auto& c : candidates_) {
        if (c.ndt_score_ > options_.ndt_score_th_ * 0.8) {  // Slightly lower threshold for hypotheses
            hypothesis_manager_->AddHypothesis(c, current_frame);
            total_hypotheses_created_++;
        }
    }
    
    // Prune old/rejected hypotheses
    int active_before = hypothesis_manager_->GetActiveCount();
    hypothesis_manager_->PruneHypotheses(current_frame);
    int pruned = active_before - hypothesis_manager_->GetActiveCount();
    total_hypotheses_rejected_ += pruned;
    
    if (options_.verbose_) {
        LOG(INFO) << "[Multi-Hyp] Frame " << current_frame 
                  << " active=" << hypothesis_manager_->GetActiveCount()
                  << " created=" << total_hypotheses_created_
                  << " committed=" << total_hypotheses_committed_
                  << " rejected=" << total_hypotheses_rejected_;
    }
}

void LoopClosing::CommitHighConfidenceLoops() {
    if (!hypothesis_manager_) return;
    
    auto committable = hypothesis_manager_->GetCommittableHypotheses();
    
    for (const auto& hyp : committable) {
        // Add as a confirmed loop closure candidate
        candidates_.push_back(hyp);
        total_hypotheses_committed_++;
        
        if (options_.verbose_) {
            LOG(INFO) << "[Multi-Hyp] COMMITTED loop " << hyp.idx1_ << " <-> " << hyp.idx2_
                      << " confidence=" << hyp.confidence_
                      << " validations=" << hyp.validation_count_;
        }
    }
}

void LoopClosing::PoseOptimization() {
    auto v = std::make_shared<miao::VertexSE3>();
    v->SetId(cur_kf_->GetID());
    v->SetEstimate(cur_kf_->GetOptPose());

    optimizer_->AddVertex(v);
    kf_vert_.emplace_back(v);

    /// 上一个关键帧的运动约束
    /// TODO 3D激光最好是跟前面多个帧都有关联

    for (int i = 1; i < 3; i++) {
        int id = cur_kf_->GetID() - i;
        if (id >= 0) {
            auto last_kf = all_keyframes_[id];
            auto e = std::make_shared<miao::EdgeSE3>();
            e->SetVertex(0, optimizer_->GetVertex(last_kf->GetID()));
            e->SetVertex(1, v);

            SE3 motion = last_kf->GetLIOPose().inverse() * cur_kf_->GetLIOPose();
            e->SetMeasurement(motion);
            e->SetInformation(info_motion_);
            optimizer_->AddEdge(e);
        }
    }

    if (options_.with_height_) {
        /// 高度约束
        auto e = std::make_shared<miao::EdgeHeightPrior>();
        e->SetVertex(0, v);
        e->SetMeasurement(0);
        e->SetInformation(Mat1d::Identity() * 1.0 / (options_.height_noise_ * options_.height_noise_));
        optimizer_->AddEdge(e);
    }
    
    // Multi-Hypothesis: Commit high-confidence loops before adding constraints
    if (options_.use_multi_hypothesis_) {
        CommitHighConfidenceLoops();
    }

    /// 回环的约束
    for (auto& c : candidates_) {
        auto e = std::make_shared<miao::EdgeSE3>();
        e->SetVertex(0, optimizer_->GetVertex(c.idx1_));
        e->SetVertex(1, optimizer_->GetVertex(c.idx2_));
        e->SetMeasurement(c.Tij_);
        e->SetInformation(info_loops_);

        auto rk = std::make_shared<miao::RobustKernelCauchy>();
        rk->SetDelta(options_.rk_loop_th_);
        e->SetRobustKernel(rk);

        optimizer_->AddEdge(e);
        edge_loops_.emplace_back(e);
        
        // Notify UI of new loop closure (only if not already notified)
        if (loop_added_cb_) {
            auto loop_pair = std::make_pair(std::min(c.idx1_, c.idx2_), std::max(c.idx1_, c.idx2_));
            if (notified_loops_.find(loop_pair) == notified_loops_.end()) {
                notified_loops_.insert(loop_pair);
                loop_added_cb_(c.idx1_, c.idx2_);
            }
        }
    }

    if (optimizer_->GetEdges().empty()) {
        return;
    }

    if (candidates_.empty()) {
        return;
    }

    optimizer_->InitializeOptimization();
    optimizer_->SetVerbose(false);

    optimizer_->Optimize(20);

    /// remove outliers
    int cnt_outliers = 0;
    for (auto& e : edge_loops_) {
        if (e->GetRobustKernel() == nullptr) {
            continue;
        }

        if (e->Chi2() > e->GetRobustKernel()->Delta()) {
            e->SetLevel(1);
            cnt_outliers++;
        } else {
            e->SetRobustKernel(nullptr);
        }
    }

    if (options_.verbose_) {
        LOG(INFO) << "loop outliers: " << cnt_outliers << "/" << edge_loops_.size();
    }

    /// get results
    for (auto& vert : kf_vert_) {
        SE3 pose = vert->Estimate();
        all_keyframes_[vert->GetId()]->SetOptPose(pose);
    }

    if (loop_cb_) {
        loop_cb_();
    }

    LOG(INFO) << "optimize finished, loops: " << edge_loops_.size();

    // LOG(INFO) << "lc: cur kf " << cur_kf_->GetID() << ", opt: " << cur_kf_->GetOptPose().translation().transpose()
    //           << ", lio: " << cur_kf_->GetLIOPose().translation().transpose();
}

}  // namespace lightning