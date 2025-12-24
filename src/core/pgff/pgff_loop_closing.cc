//
// PGFF-Enhanced Loop Closing Implementation
//

#include "core/pgff/pgff_loop_closing.h"
#include "utils/pointcloud_utils.h"

#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

#include <glog/logging.h>
#include <algorithm>
#include <cmath>

namespace lightning {
namespace pgff {

// Use the VoxelGrid helper from pointcloud_utils
using lightning::VoxelGrid;

bool PGFFLoopClosing::ProcessKeyframe(Keyframe::Ptr kf, double surprise_score) {
    if (!kf) return false;
    
    // Store keyframe
    all_keyframes_.push_back(kf);
    frames_since_last_loop_++;
    
    // Update surprise statistics
    UpdateSurpriseState(surprise_score);
    
    // Determine if we should search for loops
    bool should_search = false;
    
    // Method 1: Periodic check
    if (frames_since_last_loop_ >= options_.loop_kf_gap) {
        should_search = true;
    }
    
    // Method 2: Surprise-triggered (PGFF enhancement)
    if (surprise_history_.size() >= options_.min_surprise_history) {
        // Low surprise = revisiting known area
        if (surprise_state_.is_drop) {
            if (options_.verbose) {
                LOG(INFO) << "[PGFF-LC] Surprise DROP detected at KF " << kf->GetID() 
                          << " (normalized: " << surprise_state_.normalized_surprise << ")";
            }
            should_search = true;
        }
        
        // High surprise spike = significant drift, need correction
        if (surprise_state_.is_spike && frames_since_last_loop_ > options_.loop_kf_gap / 2) {
            if (options_.verbose) {
                LOG(INFO) << "[PGFF-LC] Surprise SPIKE detected at KF " << kf->GetID()
                          << " (normalized: " << surprise_state_.normalized_surprise << ")";
            }
            should_search = true;
        }
    }
    
    if (!should_search) {
        return false;
    }
    
    // Minimum keyframe requirement
    if (all_keyframes_.size() < options_.closest_id_th + 10) {
        return false;
    }
    
    // Detect candidates based on geometry
    auto candidates = DetectLoopCandidates(kf);
    
    if (candidates.empty()) {
        return false;
    }
    
    if (options_.verbose) {
        LOG(INFO) << "[PGFF-LC] Found " << candidates.size() << " loop candidates for KF " << kf->GetID();
    }
    
    // Verify candidates using NDT/ICP
    auto verified = VerifyLoopCandidates(candidates, kf);
    
    if (verified.empty()) {
        if (options_.verbose) {
            LOG(INFO) << "[PGFF-LC] No candidates passed verification";
        }
        return false;
    }
    
    if (options_.verbose) {
        LOG(INFO) << "[PGFF-LC] " << verified.size() << " loops verified!";
    }
    
    // Store verified loops
    for (const auto& loop : verified) {
        verified_loops_.push_back(loop);
    }
    
    // Optimize pose graph
    OptimizePoseGraph(verified);
    
    // Update state
    last_loop_kf_ = kf;
    frames_since_last_loop_ = 0;
    
    // Callback
    if (loop_callback_) {
        loop_callback_(verified);
    }
    
    return true;
}

void PGFFLoopClosing::UpdateSurpriseState(double surprise) {
    surprise_history_.push_back(surprise);
    
    // Keep limited history
    while (surprise_history_.size() > 100) {
        surprise_history_.pop_front();
    }
    
    if (surprise_history_.size() < 5) {
        surprise_state_.current_surprise = surprise;
        return;
    }
    
    // Compute running statistics
    double sum = 0, sum_sq = 0;
    for (double s : surprise_history_) {
        sum += s;
        sum_sq += s * s;
    }
    
    double n = surprise_history_.size();
    double mean = sum / n;
    double variance = (sum_sq / n) - (mean * mean);
    double std_dev = std::sqrt(std::max(variance, 1e-6));
    
    surprise_state_.current_surprise = surprise;
    surprise_state_.running_mean = mean;
    surprise_state_.running_std = std_dev;
    surprise_state_.normalized_surprise = (surprise - mean) / std_dev;
    
    // Detect anomalies
    // DROP: surprise significantly below average (revisiting known area)
    surprise_state_.is_drop = (surprise < mean * options_.surprise_drop_threshold) ||
                              (surprise_state_.normalized_surprise < -1.5);
    
    // SPIKE: surprise significantly above average (drift/new area)
    surprise_state_.is_spike = (surprise_state_.normalized_surprise > options_.surprise_spike_threshold);
}

std::vector<LoopCandidate> PGFFLoopClosing::DetectLoopCandidates(Keyframe::Ptr current_kf) {
    std::vector<LoopCandidate> candidates;
    
    Vec3d current_pos = current_kf->GetOptPose().translation();
    int current_id = static_cast<int>(current_kf->GetID());
    
    Keyframe::Ptr last_checked = nullptr;
    
    for (auto& kf : all_keyframes_) {
        int kf_id = static_cast<int>(kf->GetID());
        
        // Skip if too close in ID (same trajectory segment)
        if (std::abs(kf_id - current_id) < options_.closest_id_th) {
            continue;
        }
        
        // Skip if too close to last checked candidate
        if (last_checked && std::abs(kf_id - static_cast<int>(last_checked->GetID())) < options_.min_id_interval) {
            continue;
        }
        
        // Check distance
        Vec3d kf_pos = kf->GetOptPose().translation();
        double dist = (kf_pos - current_pos).head<2>().norm();  // 2D distance
        
        if (dist < options_.min_range || dist > options_.max_range) {
            continue;
        }
        
        // Create candidate
        LoopCandidate c(kf_id, current_id);
        c.Tij_ = kf->GetLIOPose().inverse() * current_kf->GetLIOPose();
        candidates.push_back(c);
        
        last_checked = kf;
    }
    
    return candidates;
}

std::vector<LoopCandidate> PGFFLoopClosing::VerifyLoopCandidates(
    const std::vector<LoopCandidate>& candidates,
    Keyframe::Ptr current_kf) {
    
    std::vector<LoopCandidate> verified;
    
    for (auto candidate : candidates) {
        // Build submaps
        CloudPtr submap1 = BuildSubmap(candidate.idx1_, true);  // World frame
        CloudPtr submap2 = current_kf->GetCloud();
        
        if (!submap1 || submap1->empty() || !submap2 || submap2->empty()) {
            continue;
        }
        
        // Initial guess from LIO poses
        Mat4f init_guess = current_kf->GetLIOPose().matrix().cast<float>();
        
        // Multi-resolution NDT
        CloudPtr output(new PointCloudType);
        std::vector<double> resolutions = {5.0, 2.0, 1.0, 0.5};
        double best_score = 0;
        
        for (double res : resolutions) {
            pcl::NormalDistributionsTransform<PointType, PointType> ndt;
            ndt.setTransformationEpsilon(0.01);
            ndt.setStepSize(0.5);
            ndt.setMaximumIterations(30);
            ndt.setResolution(res);
            
            CloudPtr filtered1 = VoxelGrid(submap1, res * 0.1);
            CloudPtr filtered2 = VoxelGrid(submap2, res * 0.1);
            
            ndt.setInputTarget(filtered1);
            ndt.setInputSource(filtered2);
            ndt.align(*output, init_guess);
            
            init_guess = ndt.getFinalTransformation();
            best_score = ndt.getTransformationProbability();
        }
        
        // Check NDT score
        if (best_score < options_.ndt_score_th) {
            if (options_.verbose) {
                LOG(INFO) << "[PGFF-LC] Candidate " << candidate.idx1_ << "->" << candidate.idx2_ 
                          << " rejected (NDT score: " << best_score << ")";
            }
            continue;
        }
        
        // Fine ICP refinement
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(2.0);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        
        CloudPtr filtered1 = VoxelGrid(submap1, 0.3);
        CloudPtr filtered2 = VoxelGrid(submap2, 0.3);
        
        icp.setInputTarget(filtered1);
        icp.setInputSource(filtered2);
        
        CloudPtr icp_output(new PointCloudType);
        icp.align(*icp_output, init_guess);
        
        double icp_fitness = icp.getFitnessScore();
        
        if (icp_fitness > options_.icp_fitness_th) {
            if (options_.verbose) {
                LOG(INFO) << "[PGFF-LC] Candidate " << candidate.idx1_ << "->" << candidate.idx2_
                          << " rejected (ICP fitness: " << icp_fitness << ")";
            }
            continue;
        }
        
        // Update candidate with refined transformation
        Mat4d T = icp.getFinalTransformation().cast<double>();
        Quatd q(T.block<3, 3>(0, 0));
        q.normalize();
        Vec3d t = T.block<3, 1>(0, 3);
        
        auto kf1 = all_keyframes_[candidate.idx1_];
        candidate.Tij_ = kf1->GetLIOPose().inverse() * SE3(q, t);
        candidate.ndt_score_ = best_score;
        
        if (options_.verbose) {
            LOG(INFO) << "[PGFF-LC] Loop VERIFIED: " << candidate.idx1_ << "->" << candidate.idx2_
                      << " (NDT: " << best_score << ", ICP: " << icp_fitness << ")";
        }
        
        verified.push_back(candidate);
    }
    
    return verified;
}

CloudPtr PGFFLoopClosing::BuildSubmap(int center_kf_id, bool in_world_frame) {
    CloudPtr submap(new PointCloudType);
    
    int half_size = options_.submap_size / 2;
    
    for (int offset = -half_size; offset <= half_size; offset += 2) {
        int kf_id = center_kf_id + offset;
        
        if (kf_id < 0 || kf_id >= static_cast<int>(all_keyframes_.size())) {
            continue;
        }
        
        auto kf = all_keyframes_[kf_id];
        CloudPtr cloud = kf->GetCloud();
        
        if (!cloud || cloud->empty()) {
            continue;
        }
        
        SE3 pose = kf->GetLIOPose();
        
        if (!in_world_frame) {
            pose = all_keyframes_[center_kf_id]->GetLIOPose().inverse() * pose;
        }
        
        CloudPtr transformed(new PointCloudType);
        pcl::transformPointCloud(*cloud, *transformed, pose.matrix());
        
        *submap += *transformed;
    }
    
    // Downsample result
    return VoxelGrid(submap, 0.2);
}

void PGFFLoopClosing::OptimizePoseGraph(const std::vector<LoopCandidate>& loops) {
    if (loops.empty() || all_keyframes_.size() < 2) {
        return;
    }
    
    // Simple pose graph optimization using Gauss-Newton
    // For production, integrate with miao optimizer
    
    LOG(INFO) << "[PGFF-LC] Optimizing pose graph with " << loops.size() << " loop constraints";
    
    // Build information matrices
    Mat6d info_motion = Mat6d::Identity();
    info_motion.block<3, 3>(0, 0) *= 100.0;  // Translation weight
    info_motion.block<3, 3>(3, 3) *= 1000.0; // Rotation weight
    
    Mat6d info_loop = Mat6d::Identity();
    info_loop.block<3, 3>(0, 0) *= 25.0;     // Loop translation weight
    info_loop.block<3, 3>(3, 3) *= 250.0;    // Loop rotation weight
    
    // Store original poses for interpolation
    std::vector<SE3> original_poses;
    for (auto& kf : all_keyframes_) {
        original_poses.push_back(kf->GetOptPose());
    }
    
    // Simple correction: distribute error along trajectory
    for (const auto& loop : loops) {
        int idx1 = loop.idx1_;
        int idx2 = loop.idx2_;
        
        if (idx1 >= idx2 || idx2 >= static_cast<int>(all_keyframes_.size())) {
            continue;
        }
        
        // Compute pose error
        SE3 T1 = all_keyframes_[idx1]->GetOptPose();
        SE3 T2 = all_keyframes_[idx2]->GetOptPose();
        
        SE3 measured_T12 = loop.Tij_;
        SE3 current_T12 = T1.inverse() * T2;
        
        // Error in se3
        SE3 error = measured_T12.inverse() * current_T12;
        Vec6d error_vec = error.log();
        
        // Distribute correction across trajectory segment
        int segment_length = idx2 - idx1;
        
        for (int i = idx1 + 1; i <= idx2; ++i) {
            double alpha = static_cast<double>(i - idx1) / segment_length;
            
            // Interpolated correction
            Vec6d correction = alpha * error_vec;
            SE3 delta = SE3::exp(-correction);  // Negative to correct
            
            SE3 original = all_keyframes_[i]->GetOptPose();
            SE3 corrected = T1 * delta * (T1.inverse() * original);
            
            all_keyframes_[i]->SetOptPose(corrected);
        }
    }
    
    LOG(INFO) << "[PGFF-LC] Pose graph optimization complete";
}

}  // namespace pgff
}  // namespace lightning
