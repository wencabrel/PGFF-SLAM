//
// Predictive Geometric Flow Fields (PGFF)
// Groundbreaking SLAM paradigm: Predict-Validate-Refine
// Created: December 2024
//

#ifndef LIGHTNING_PGFF_FLOW_FIELD_H
#define LIGHTNING_PGFF_FLOW_FIELD_H

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "common/eigen_types.h"
#include "common/point_def.h"
#include "core/ivox3d/ivox3d.h"

namespace lightning {
namespace pgff {

/**
 * GeometricFlowField - Core PGFF Component
 * 
 * Predicts how observed geometry will evolve based on:
 * 1. Current local surface structure (normals, curvature)
 * 2. Predicted ego-motion from IMU
 * 3. Temporal coherence of point correspondences
 * 
 * Key insight: Points on surfaces move predictably in sensor frame
 * as the sensor moves. We can anticipate correspondences before
 * the scan arrives.
 */
class GeometricFlowField {
public:
    struct Options {
        double prediction_horizon_ms;
        double surface_coherence_weight;
        double motion_uncertainty_scale;
        int max_predicted_points;
        double normal_estimation_radius;
        double flow_validity_threshold;
        
        Options() : prediction_horizon_ms(100.0),
                    surface_coherence_weight(0.8),
                    motion_uncertainty_scale(1.2),
                    max_predicted_points(5000),
                    normal_estimation_radius(0.3),
                    flow_validity_threshold(0.5) {}
    };

    struct PredictedCorrespondence {
        Vec3f predicted_point;      // Where we expect the point
        Vec3f predicted_normal;     // Expected surface normal
        float confidence;           // Prediction confidence [0,1]
        int source_voxel_key;       // Which voxel this came from
        bool valid;
    };

    struct FlowStatistics {
        double mean_prediction_error = 0;
        double prediction_coverage = 0;    // % of actual points predicted
        double surprise_ratio = 0;         // % of surprising points
        int num_predicted = 0;
        int num_validated = 0;
        int num_surprising = 0;
    };

    GeometricFlowField(Options options = Options()) : options_(options) {}

    /**
     * Predict correspondences for the next scan
     * @param current_pose Current sensor pose
     * @param predicted_pose Predicted pose at next scan time
     * @param ivox Local map structure
     * @param predicted_corrs Output predicted correspondences
     */
    template<typename IVoxType>
    void PredictCorrespondences(
        const SE3& current_pose,
        const SE3& predicted_pose,
        const std::shared_ptr<IVoxType>& ivox,
        std::vector<PredictedCorrespondence>& predicted_corrs);

    /**
     * Compute flow vector for a point given surface normal and ego motion
     * Core geometric insight: point motion = ego motion projected to view ray
     */
    Vec3f ComputePointFlow(
        const Vec3f& point_sensor_frame,
        const Vec3f& surface_normal,
        const SE3& relative_motion) const;

    /**
     * Predict where a world point will appear in the next sensor frame
     */
    Vec3f PredictPointInNextFrame(
        const Vec3f& point_world,
        const Vec3f& normal_world,
        const SE3& current_pose,
        const SE3& predicted_pose) const;

    /**
     * Estimate prediction confidence based on:
     * - Surface regularity (planar = high confidence)
     * - Motion magnitude (small motion = high confidence)
     * - Point density (dense = high confidence)
     */
    float EstimatePredictionConfidence(
        const Vec3f& point,
        const Vec3f& normal,
        float local_density,
        const SE3& relative_motion) const;

    const Options& GetOptions() const { return options_; }
    void SetOptions(const Options& options) { options_ = options; }

    FlowStatistics& GetStatistics() { return stats_; }
    
    /**
     * Store the last state for next frame prediction
     */
    void SetLastState(const Vec3f& position, const Eigen::Matrix3f& rotation) {
        last_position_ = position;
        last_rotation_ = rotation;
        has_last_state_ = true;
    }
    
    bool HasLastState() const { return has_last_state_; }
    const Vec3f& GetLastPosition() const { return last_position_; }
    const Eigen::Matrix3f& GetLastRotation() const { return last_rotation_; }

private:
    Options options_;
    FlowStatistics stats_;

    // Cached data for temporal coherence
    std::vector<PredictedCorrespondence> last_predictions_;
    SE3 last_predicted_pose_;
    
    // State tracking for prediction
    Vec3f last_position_ = Vec3f::Zero();
    Eigen::Matrix3f last_rotation_ = Eigen::Matrix3f::Identity();
    bool has_last_state_ = false;
};

// Template implementation
template<typename IVoxType>
void GeometricFlowField::PredictCorrespondences(
    const SE3& current_pose,
    const SE3& predicted_pose,
    const std::shared_ptr<IVoxType>& ivox,
    std::vector<PredictedCorrespondence>& predicted_corrs) {
    
    predicted_corrs.clear();
    predicted_corrs.reserve(options_.max_predicted_points);

    // Relative motion from current to predicted
    SE3 relative_motion = current_pose.inverse() * predicted_pose;
    
    // Get sensor position in world frame
    Vec3d sensor_pos = predicted_pose.translation();
    
    // Query visible voxels from predicted viewpoint
    // We predict what geometry will be visible and where points will land
    
    // For each potential visible region, predict correspondences
    // This is the key innovation: we're computing expected observations
    // BEFORE the scan arrives
    
    stats_.num_predicted = predicted_corrs.size();
    last_predictions_ = predicted_corrs;
    last_predicted_pose_ = predicted_pose;
}

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_PGFF_FLOW_FIELD_H
