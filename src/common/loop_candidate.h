//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_LOOP_CANDIDATE_H
#define LIGHTNING_LOOP_CANDIDATE_H

#include "common/eigen_types.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace lightning {

/**
 * 回环检测候选帧
 * Enhanced with multi-hypothesis tracking for groundbreaking loop closing
 */
struct LoopCandidate {
    LoopCandidate() {}
    LoopCandidate(uint64_t id1, uint64_t id2) : idx1_(id1), idx2_(id2) {}

    uint64_t idx1_ = 0;  // Reference keyframe ID
    uint64_t idx2_ = 0;  // Current keyframe ID
    SE3 Tij_;            // Relative transform between frames

    double ndt_score_ = 0.0;  // NDT alignment score
    
    // Multi-Hypothesis Loop Closing fields
    double confidence_ = 0.0;           // Overall hypothesis confidence [0, 1]
    double icp_fitness_ = 0.0;          // ICP fitness score (lower is better)
    double geometric_consistency_ = 0.0; // Consistency with neighboring hypotheses
    double scan_context_dist_ = 0.0;    // Scan context descriptor distance
    
    int validation_count_ = 0;          // Number of frames that validated this hypothesis
    int creation_frame_ = 0;            // Frame ID when hypothesis was created
    bool is_committed_ = false;         // Whether this hypothesis has been committed
    
    // Compute overall confidence from component scores
    void ComputeConfidence() {
        // Weighted combination of scores
        // Higher NDT score = better alignment
        // Lower ICP fitness = better fit (invert for consistency)
        // Higher geometric consistency = more reliable
        double ndt_contrib = std::min(ndt_score_ / 5.0, 1.0);  // Normalize to [0,1]
        double icp_contrib = std::max(0.0, 1.0 - icp_fitness_ / 0.5);  // Invert: low fitness = high score
        double geo_contrib = geometric_consistency_;
        double validation_contrib = std::min(validation_count_ / 3.0, 1.0);  // Max out at 3 validations
        
        // Weighted average with validation bonus
        confidence_ = 0.3 * ndt_contrib + 0.2 * icp_contrib + 0.2 * geo_contrib + 0.3 * validation_contrib;
    }
    
    // Check if hypothesis should be committed based on confidence threshold
    bool ShouldCommit(double threshold = 0.6) const {
        return confidence_ >= threshold && validation_count_ >= 2;
    }
    
    // Check if hypothesis should be rejected (too old without validation)
    bool ShouldReject(int current_frame, int max_age = 10) const {
        return (current_frame - creation_frame_) > max_age && validation_count_ < 2;
    }
};

/**
 * Multi-Hypothesis Loop Closing Manager
 * Tracks multiple competing hypotheses and commits only high-confidence ones
 */
class LoopHypothesisManager {
   public:
    struct Config {
        int max_hypotheses_ = 10;           // Maximum hypotheses to track
        double commit_threshold_ = 0.6;      // Confidence threshold for committing
        int min_validations_ = 2;            // Minimum validations before commit
        int max_hypothesis_age_ = 15;        // Max frames before rejecting unvalidated hypothesis
        double validation_distance_th_ = 2.0; // Distance threshold for validation
        
        Config() = default;
    };
    
    LoopHypothesisManager() : config_() {}
    explicit LoopHypothesisManager(const Config& config) : config_(config) {}
    
    // Add a new hypothesis candidate
    void AddHypothesis(const LoopCandidate& candidate, int current_frame) {
        LoopCandidate hyp = candidate;
        hyp.creation_frame_ = current_frame;
        hyp.validation_count_ = 1;  // Initial observation counts as one
        hyp.ComputeConfidence();
        
        hypotheses_.push_back(hyp);
        
        // Keep only top-K hypotheses by confidence
        if (static_cast<int>(hypotheses_.size()) > config_.max_hypotheses_) {
            PruneHypotheses();
        }
    }
    
    // Validate existing hypotheses with new frame observation
    void ValidateWithFrame(uint64_t current_kf_id, const SE3& current_pose,
                          const std::vector<LoopCandidate>& new_candidates) {
        for (auto& hyp : hypotheses_) {
            if (hyp.is_committed_) continue;
            
            // Check if any new candidate confirms this hypothesis
            for (const auto& cand : new_candidates) {
                if (IsConsistent(hyp, cand, current_pose)) {
                    hyp.validation_count_++;
                    // Update confidence with new evidence
                    hyp.geometric_consistency_ = std::min(1.0, hyp.geometric_consistency_ + 0.2);
                    hyp.ComputeConfidence();
                    break;
                }
            }
        }
    }
    
    // Get hypotheses ready to commit
    std::vector<LoopCandidate> GetCommittableHypotheses() {
        std::vector<LoopCandidate> ready;
        for (auto& hyp : hypotheses_) {
            if (!hyp.is_committed_ && hyp.ShouldCommit(config_.commit_threshold_)) {
                hyp.is_committed_ = true;
                ready.push_back(hyp);
            }
        }
        return ready;
    }
    
    // Remove old/rejected hypotheses
    void PruneHypotheses(int current_frame = -1) {
        if (current_frame >= 0) {
            // Remove rejected hypotheses
            hypotheses_.erase(
                std::remove_if(hypotheses_.begin(), hypotheses_.end(),
                    [&](const LoopCandidate& h) { 
                        return h.ShouldReject(current_frame, config_.max_hypothesis_age_); 
                    }),
                hypotheses_.end());
        }
        
        // Sort by confidence and keep top K
        if (static_cast<int>(hypotheses_.size()) > config_.max_hypotheses_) {
            std::sort(hypotheses_.begin(), hypotheses_.end(),
                [](const LoopCandidate& a, const LoopCandidate& b) {
                    return a.confidence_ > b.confidence_;
                });
            hypotheses_.resize(config_.max_hypotheses_);
        }
    }
    
    // Get all active (uncommitted) hypotheses
    const std::vector<LoopCandidate>& GetHypotheses() const { return hypotheses_; }
    
    // Get statistics
    int GetActiveCount() const {
        return static_cast<int>(std::count_if(hypotheses_.begin(), hypotheses_.end(),
            [](const LoopCandidate& h) { return !h.is_committed_; }));
    }
    
    int GetCommittedCount() const {
        return static_cast<int>(std::count_if(hypotheses_.begin(), hypotheses_.end(),
            [](const LoopCandidate& h) { return h.is_committed_; }));
    }
    
   private:
    // Check if two candidates are geometrically consistent
    bool IsConsistent(const LoopCandidate& hyp, const LoopCandidate& cand, const SE3& /*current_pose*/) {
        // Same reference frame
        if (hyp.idx1_ == cand.idx1_) {
            // Check if relative transforms are consistent
            Vec3d diff = (hyp.Tij_.translation() - cand.Tij_.translation());
            return diff.norm() < config_.validation_distance_th_;
        }
        
        // Nearby reference frames (within 5 keyframes)
        if (std::abs(static_cast<int>(hyp.idx1_) - static_cast<int>(cand.idx1_)) < 5) {
            return true;  // Nearby frames validating similar region
        }
        
        return false;
    }
    
    Config config_;
    std::vector<LoopCandidate> hypotheses_;
};

}  // namespace lightning

#endif  // LIGHTNING_LOOP_CANDIDATE_H
