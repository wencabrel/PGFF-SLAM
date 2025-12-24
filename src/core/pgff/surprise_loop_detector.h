//
// Predictive Geometric Flow Fields (PGFF)
// Surprise-Driven Loop Detection
//

#ifndef LIGHTNING_PGFF_SURPRISE_LOOP_DETECTOR_H
#define LIGHTNING_PGFF_SURPRISE_LOOP_DETECTOR_H

#include <vector>
#include <deque>

#include "common/keyframe.h"
#include "core/pgff/predictive_lio.h"

namespace lightning {
namespace pgff {

/**
 * SurpriseLoopDetector - Implicit Loop Closure from Prediction Failures
 * 
 * Key insight: When we revisit a place, our predictions suddenly become
 * very wrong because:
 * 1. The stored map geometry differs from current observations
 * 2. Drift has accumulated, causing prediction errors
 * 
 * This provides INSTANT loop closure hints without periodic scanning.
 * Traditional methods check every N keyframes; PGFF detects anomalies
 * in real-time.
 */
class SurpriseLoopDetector {
public:
    struct Options {
        // Surprise thresholds for loop hint
        double surprise_spike_threshold;
        double min_surprise_for_loop;
        
        // Temporal filtering
        int surprise_history_size;
        double surprise_spike_duration;
        
        // Loop candidate criteria
        double min_distance_for_loop;
        int min_keyframes_for_loop;
        
        bool verbose;
        
        Options() : surprise_spike_threshold(2.5),
                    min_surprise_for_loop(0.8),
                    surprise_history_size(50),
                    surprise_spike_duration(3),
                    min_distance_for_loop(20.0),
                    min_keyframes_for_loop(30),
                    verbose(false) {}
    };

    struct LoopHint {
        int keyframe_id;
        double surprise_score;
        double confidence;
        Vec3d position;
        bool triggered;
    };

    SurpriseLoopDetector(Options options = Options()) : options_(options) {}

    /**
     * Update with current frame surprise
     * Returns true if loop closure should be triggered
     */
    bool Update(Keyframe::Ptr kf, double frame_surprise) {
        surprise_history_.push_back(frame_surprise);
        if (surprise_history_.size() > options_.surprise_history_size) {
            surprise_history_.pop_front();
        }
        
        // Compute running statistics
        double mean = 0, var = 0;
        for (double s : surprise_history_) mean += s;
        mean /= surprise_history_.size();
        
        for (double s : surprise_history_) var += (s - mean) * (s - mean);
        var = std::sqrt(var / surprise_history_.size());
        
        running_mean_ = mean;
        running_std_ = var;
        
        // Detect surprise spike
        bool is_spike = (frame_surprise > mean + options_.surprise_spike_threshold * var) &&
                        (frame_surprise > options_.min_surprise_for_loop);
        
        if (is_spike) {
            spike_counter_++;
        } else {
            spike_counter_ = 0;
        }
        
        // Check if we should trigger loop search
        bool should_trigger = false;
        
        if (spike_counter_ >= options_.surprise_spike_duration) {
            // Sustained surprise spike - potential loop!
            
            // Check distance criterion
            if (last_loop_kf_) {
                double distance = (kf->GetOptPose().translation() - 
                                  last_loop_kf_->GetOptPose().translation()).norm();
                int kf_gap = kf->GetID() - last_loop_kf_->GetID();
                
                if (distance > options_.min_distance_for_loop &&
                    kf_gap > options_.min_keyframes_for_loop) {
                    should_trigger = true;
                }
            } else if (kf->GetID() > options_.min_keyframes_for_loop) {
                should_trigger = true;
            }
        }
        
        if (should_trigger) {
            last_hint_.keyframe_id = kf->GetID();
            last_hint_.surprise_score = frame_surprise;
            last_hint_.confidence = (frame_surprise - mean) / (var + 1e-6);
            last_hint_.position = kf->GetOptPose().translation();
            last_hint_.triggered = true;
            
            last_loop_kf_ = kf;
            spike_counter_ = 0;
            
            if (options_.verbose) {
                LOG(INFO) << "[PGFF-Loop] Surprise spike detected at KF " 
                          << kf->GetID() << " (score: " << frame_surprise << ")";
            }
        }
        
        return should_trigger;
    }

    /**
     * Get the most recent loop hint
     */
    const LoopHint& GetLastHint() const { return last_hint_; }
    
    /**
     * Get current surprise statistics
     */
    double GetRunningSurpriseMean() const { return running_mean_; }
    double GetRunningSurpriseStd() const { return running_std_; }
    
    void Reset() {
        surprise_history_.clear();
        spike_counter_ = 0;
        last_loop_kf_ = nullptr;
        running_mean_ = 0;
        running_std_ = 0;
    }

private:
    Options options_;
    
    std::deque<double> surprise_history_;
    int spike_counter_ = 0;
    double running_mean_ = 0;
    double running_std_ = 0;
    
    Keyframe::Ptr last_loop_kf_ = nullptr;
    LoopHint last_hint_;
};

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_PGFF_SURPRISE_LOOP_DETECTOR_H
