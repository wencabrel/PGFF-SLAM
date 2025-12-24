//
// Predictive Geometric Flow Fields (PGFF)
// Predictive LIO - Drop-in enhancement for LaserMapping
//

#ifndef LIGHTNING_PGFF_PREDICTIVE_LIO_H
#define LIGHTNING_PGFF_PREDICTIVE_LIO_H

#include <memory>
#include <vector>

#include "common/eigen_types.h"
#include "common/nav_state.h"
#include "common/point_def.h"
#include "core/pgff/flow_field.h"
#include "core/pgff/jacobian_cache.h"
#include "core/pgff/surprise_detector.h"

namespace lightning {
namespace pgff {

/**
 * PredictiveLIO - PGFF Enhancement Layer
 * 
 * This class wraps around the existing LIO pipeline to add
 * predictive processing capabilities WITHOUT modifying the base code.
 * 
 * Usage:
 *   1. Create PredictiveLIO instance alongside LaserMapping
 *   2. Call PreparePrediction() before each scan
 *   3. Call SelectPointsToProcess() to get surprising points
 *   4. Use GetCachedJacobian() for non-surprising points
 *   5. Call UpdateWithResults() after ESKF update
 * 
 * The existing LaserMapping code remains untouched - this is
 * a pure enhancement layer.
 */
class PredictiveLIO {
public:
    struct Options {
        bool enabled;
        
        // Sub-module options
        GeometricFlowField::Options flow_options;
        JacobianCache::Options cache_options;
        SurpriseDetector::Options surprise_options;
        
        // Performance tuning
        double min_prediction_confidence;
        bool use_jacobian_cache;
        bool use_selective_processing;
        bool verbose;
        
        Options() : enabled(true),
                    flow_options(),
                    cache_options(),
                    surprise_options(),
                    min_prediction_confidence(0.3),
                    use_jacobian_cache(true),
                    use_selective_processing(true),
                    verbose(false) {}
    };

    struct Statistics {
        // Per-frame statistics
        int total_points = 0;
        int surprising_points = 0;
        int cached_points = 0;
        double prediction_accuracy = 0;
        double time_saved_ms = 0;
        
        // Cumulative statistics
        long total_frames = 0;
        long total_points_processed = 0;
        long total_points_skipped = 0;
        double cumulative_time_saved_ms = 0;
        double average_surprise_ratio = 0;
    };

    PredictiveLIO(Options options = Options());

    /**
     * Phase 1: Prepare predictions before scan arrives
     * Uses IMU-propagated state to predict observations
     */
    void PreparePrediction(
        const NavState& current_state,
        const NavState& predicted_state,
        const SE3& predicted_pose);

    /**
     * Phase 2: When scan arrives, compute surprise and select points
     * Returns indices of points that should be fully processed
     */
    std::vector<int> SelectPointsToProcess(
        const CloudPtr& scan,
        const std::vector<float>& residuals);

    /**
     * Phase 3: Get cached Jacobian rows for non-surprising points
     * These can be directly used in the ESKF update
     */
    bool GetCachedJacobianRows(
        const std::vector<int>& non_surprising_indices,
        Eigen::MatrixXd& H_cached,
        Eigen::VectorXd& residuals_cached);

    /**
     * Phase 4: Update internal state after ESKF completes
     */
    void UpdateWithResults(
        const NavState& updated_state,
        const Eigen::MatrixXd& H_full,
        const std::vector<float>& residuals,
        const std::vector<bool>& point_validity);

    /**
     * Predict residuals for next frame (for surprise computation)
     */
    void PredictResiduals(
        const CloudPtr& scan,
        const NavState& predicted_state,
        std::vector<float>& predicted_residuals);

    /**
     * Check if a specific point should be processed
     */
    bool ShouldProcessPoint(int index) const;

    /**
     * Get the surprise score for a point
     */
    float GetPointSurprise(int index) const;

    // Accessors
    const Statistics& GetStatistics() const { return stats_; }
    void ResetStatistics() { stats_ = Statistics(); }
    
    const Options& GetOptions() const { return options_; }
    void SetOptions(const Options& options) { options_ = options; }
    
    bool IsEnabled() const { return options_.enabled; }
    void SetEnabled(bool enabled) { options_.enabled = enabled; }

    // Access to sub-modules for fine-grained control
    GeometricFlowField& GetFlowField() { return flow_field_; }
    JacobianCache& GetJacobianCache() { return jacobian_cache_; }
    SurpriseDetector& GetSurpriseDetector() { return surprise_detector_; }

private:
    void UpdateStatistics(int total, int surprising, int cached);
    
    Options options_;
    Statistics stats_;
    
    // Sub-modules
    GeometricFlowField flow_field_;
    JacobianCache jacobian_cache_;
    SurpriseDetector surprise_detector_;
    
    // Current frame state
    std::vector<SurpriseDetector::PointSurprise> current_surprises_;
    std::vector<int> points_to_process_;
    std::vector<float> predicted_residuals_;
    
    // State tracking
    NavState last_state_;
    bool has_prediction_ = false;
};

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_PGFF_PREDICTIVE_LIO_H
