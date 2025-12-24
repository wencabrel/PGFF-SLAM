//
// Predictive Geometric Flow Fields (PGFF)
// Predictive LIO Implementation
//

#include "core/pgff/predictive_lio.h"
#include <glog/logging.h>
#include <chrono>

namespace lightning {
namespace pgff {

PredictiveLIO::PredictiveLIO(Options options)
    : options_(options),
      flow_field_(options.flow_options),
      jacobian_cache_(options.cache_options),
      surprise_detector_(options.surprise_options) {
    
    if (options_.verbose) {
        LOG(INFO) << "[PGFF] Predictive LIO initialized";
        LOG(INFO) << "[PGFF] Selective processing: " 
                  << (options_.use_selective_processing ? "ON" : "OFF");
        LOG(INFO) << "[PGFF] Jacobian caching: " 
                  << (options_.use_jacobian_cache ? "ON" : "OFF");
    }
}

void PredictiveLIO::PreparePrediction(
    const NavState& current_state,
    const NavState& predicted_state,
    const SE3& predicted_pose) {
    
    if (!options_.enabled) return;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Store predicted state for later comparison
    last_state_ = current_state;
    
    // Clear previous frame data
    current_surprises_.clear();
    points_to_process_.clear();
    predicted_residuals_.clear();
    
    // The flow field will be used when the scan arrives
    // For now, just mark that we have a prediction ready
    has_prediction_ = true;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (options_.verbose) {
        LOG(INFO) << "[PGFF] Prediction prepared in " << elapsed << " ms";
    }
}

std::vector<int> PredictiveLIO::SelectPointsToProcess(
    const CloudPtr& scan,
    const std::vector<float>& residuals) {
    
    if (!options_.enabled || !options_.use_selective_processing) {
        // Return all points
        std::vector<int> all_points(scan->size());
        std::iota(all_points.begin(), all_points.end(), 0);
        stats_.total_points = scan->size();
        stats_.surprising_points = scan->size();
        return all_points;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    stats_.total_points = scan->size();
    
    // Quick surprise computation from residuals
    std::vector<bool> is_surprising = surprise_detector_.QuickSurpriseFromResiduals(
        residuals, predicted_residuals_);
    
    // Build list of surprising points
    points_to_process_.clear();
    for (size_t i = 0; i < is_surprising.size(); i++) {
        if (is_surprising[i]) {
            points_to_process_.push_back(i);
        }
    }
    
    stats_.surprising_points = points_to_process_.size();
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (options_.verbose) {
        LOG(INFO) << "[PGFF] Selected " << points_to_process_.size() 
                  << "/" << scan->size() << " surprising points in " 
                  << elapsed << " ms";
    }
    
    return points_to_process_;
}

bool PredictiveLIO::GetCachedJacobianRows(
    const std::vector<int>& non_surprising_indices,
    Eigen::MatrixXd& H_cached,
    Eigen::VectorXd& residuals_cached) {
    
    if (!options_.enabled || !options_.use_jacobian_cache) {
        return false;
    }
    
    if (!jacobian_cache_.isValid(last_state_)) {
        return false;
    }
    
    const auto& H_full = jacobian_cache_.getCachedH();
    
    if (H_full.rows() == 0) {
        return false;
    }
    
    // Extract cached rows for non-surprising points
    int num_rows = std::min(static_cast<int>(non_surprising_indices.size()),
                           static_cast<int>(H_full.rows()));
    
    H_cached.resize(num_rows, H_full.cols());
    residuals_cached.resize(num_rows);
    
    for (int i = 0; i < num_rows; i++) {
        int idx = non_surprising_indices[i];
        if (idx < H_full.rows()) {
            H_cached.row(i) = H_full.row(idx);
            if (idx < predicted_residuals_.size()) {
                residuals_cached(i) = predicted_residuals_[idx];
            }
        }
    }
    
    stats_.cached_points = num_rows;
    
    if (options_.verbose) {
        LOG(INFO) << "[PGFF] Retrieved " << num_rows << " cached Jacobian rows";
    }
    
    return true;
}

void PredictiveLIO::UpdateWithResults(
    const NavState& updated_state,
    const Eigen::MatrixXd& H_full,
    const std::vector<float>& residuals,
    const std::vector<bool>& point_validity) {
    
    if (!options_.enabled) return;
    
    // Update Jacobian cache
    if (options_.use_jacobian_cache && H_full.rows() > 0) {
        jacobian_cache_.updateCache(H_full, updated_state, point_validity);
    }
    
    // Store residuals for next frame prediction
    predicted_residuals_ = residuals;
    
    // Update statistics
    UpdateStatistics(stats_.total_points, stats_.surprising_points, stats_.cached_points);
    
    last_state_ = updated_state;
    has_prediction_ = false;
}

void PredictiveLIO::PredictResiduals(
    const CloudPtr& scan,
    const NavState& predicted_state,
    std::vector<float>& predicted_residuals) {
    
    // For now, use last frame's residuals as prediction
    // This works well because residuals change slowly
    predicted_residuals = predicted_residuals_;
    
    // If sizes don't match, resize with zeros
    if (predicted_residuals.size() != scan->size()) {
        predicted_residuals.resize(scan->size(), 0.0f);
    }
}

bool PredictiveLIO::ShouldProcessPoint(int index) const {
    if (!options_.enabled || !options_.use_selective_processing) {
        return true;
    }
    
    return std::binary_search(points_to_process_.begin(), 
                              points_to_process_.end(), 
                              index);
}

float PredictiveLIO::GetPointSurprise(int index) const {
    if (index < current_surprises_.size()) {
        return current_surprises_[index].surprise_score;
    }
    return 0.0f;
}

void PredictiveLIO::UpdateStatistics(int total, int surprising, int cached) {
    stats_.total_frames++;
    stats_.total_points_processed += surprising;
    stats_.total_points_skipped += (total - surprising);
    
    if (total > 0) {
        double surprise_ratio = static_cast<double>(surprising) / total;
        // Exponential moving average
        double alpha = 0.1;
        stats_.average_surprise_ratio = 
            alpha * surprise_ratio + (1 - alpha) * stats_.average_surprise_ratio;
    }
    
    // Estimate time saved (rough approximation)
    // Assuming ~0.1ms per point for ObsModel
    double time_saved = (total - surprising) * 0.0001;  // ms
    stats_.time_saved_ms = time_saved;
    stats_.cumulative_time_saved_ms += time_saved;
    
    if (options_.verbose && stats_.total_frames % 100 == 0) {
        LOG(INFO) << "[PGFF] Statistics after " << stats_.total_frames << " frames:";
        LOG(INFO) << "[PGFF]   Average surprise ratio: " 
                  << stats_.average_surprise_ratio * 100 << "%";
        LOG(INFO) << "[PGFF]   Cumulative time saved: " 
                  << stats_.cumulative_time_saved_ms << " ms";
        LOG(INFO) << "[PGFF]   Points processed: " << stats_.total_points_processed;
        LOG(INFO) << "[PGFF]   Points skipped: " << stats_.total_points_skipped;
    }
}

}  // namespace pgff
}  // namespace lightning
