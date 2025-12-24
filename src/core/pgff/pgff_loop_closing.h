//
// PGFF-Enhanced Loop Closing
// Uses prediction surprise to improve loop closure detection
//

#ifndef LIGHTNING_PGFF_LOOP_CLOSING_H
#define LIGHTNING_PGFF_LOOP_CLOSING_H

#include <deque>
#include <vector>
#include <memory>

#include "common/keyframe.h"
#include "common/loop_candidate.h"
#include "common/eigen_types.h"

namespace lightning {
namespace pgff {

/**
 * PGFFLoopClosing - Predictive Geometric Flow Fields Enhanced Loop Closure
 * 
 * Key insight: When revisiting a place, two phenomena occur:
 * 1. SURPRISE DROP: Predictions become accurate again (low surprise)
 * 2. GEOMETRIC MATCH: Point cloud alignment quality improves
 * 
 * By combining these signals, we can:
 * - Detect potential loop closures FASTER than periodic checking
 * - VERIFY loop candidates more reliably
 * - REDUCE false positives through consistency checks
 */
class PGFFLoopClosing {
public:
    struct Options {
        // Surprise-based loop triggering
        double surprise_drop_threshold;
        double surprise_spike_threshold;
        int min_surprise_history;
        
        // Traditional loop parameters
        int loop_kf_gap;
        int min_id_interval;
        int closest_id_th;
        double max_range;
        double min_range;
        
        // ICP/NDT verification
        double ndt_score_th;
        double icp_fitness_th;
        int submap_size;
        
        // PGFF consistency verification  
        bool use_pgff_verification;
        double pgff_consistency_th;
        
        bool verbose;
        
        Options() : 
            surprise_drop_threshold(0.4),
            surprise_spike_threshold(2.0),
            min_surprise_history(20),
            loop_kf_gap(15),
            min_id_interval(15),
            closest_id_th(40),
            max_range(35.0),
            min_range(5.0),
            ndt_score_th(0.8),
            icp_fitness_th(0.5),
            submap_size(30),
            use_pgff_verification(true),
            pgff_consistency_th(0.3),
            verbose(true) {}
    };

    struct SurpriseState {
        double current_surprise = 0;
        double running_mean = 0;
        double running_std = 0;
        double normalized_surprise = 0;  // (current - mean) / std
        bool is_spike = false;
        bool is_drop = false;
    };

    PGFFLoopClosing(Options options = Options()) : options_(options) {}
    
    /**
     * Update with keyframe and its PGFF surprise score
     * Returns true if loop closure optimization was performed
     */
    bool ProcessKeyframe(Keyframe::Ptr kf, double surprise_score);
    
    /**
     * Get current surprise state for debugging/visualization
     */
    const SurpriseState& GetSurpriseState() const { return surprise_state_; }
    
    /**
     * Get all keyframes (for optimization access)
     */
    std::vector<Keyframe::Ptr>& GetKeyframes() { return all_keyframes_; }
    const std::vector<Keyframe::Ptr>& GetKeyframes() const { return all_keyframes_; }
    
    /**
     * Get detected loop candidates
     */
    const std::vector<LoopCandidate>& GetLoopCandidates() const { return verified_loops_; }
    
    /**
     * Set callback for when loop closure is detected and applied
     */
    using LoopClosedCallback = std::function<void(const std::vector<LoopCandidate>&)>;
    void SetLoopClosedCallback(LoopClosedCallback cb) { loop_callback_ = cb; }

private:
    /**
     * Update surprise statistics
     */
    void UpdateSurpriseState(double surprise);
    
    /**
     * Detect loop candidates based on surprise and geometry
     */
    std::vector<LoopCandidate> DetectLoopCandidates(Keyframe::Ptr current_kf);
    
    /**
     * Verify loop candidates using NDT/ICP
     */
    std::vector<LoopCandidate> VerifyLoopCandidates(
        const std::vector<LoopCandidate>& candidates,
        Keyframe::Ptr current_kf);
    
    /**
     * Build submap around a keyframe
     */
    CloudPtr BuildSubmap(int center_kf_id, bool in_world_frame);
    
    /**
     * Perform pose graph optimization with loop constraints
     */
    void OptimizePoseGraph(const std::vector<LoopCandidate>& loops);

    Options options_;
    
    // Keyframe storage
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_loop_kf_ = nullptr;
    int frames_since_last_loop_ = 0;
    
    // Surprise tracking
    std::deque<double> surprise_history_;
    SurpriseState surprise_state_;
    
    // Loop storage
    std::vector<LoopCandidate> verified_loops_;
    
    // Callback
    LoopClosedCallback loop_callback_;
};

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_PGFF_LOOP_CLOSING_H
