//
// Predictive Geometric Flow Fields (PGFF)
// Enhanced Observation Model - Selective point processing
//

#ifndef LIGHTNING_PGFF_ENHANCED_OBS_MODEL_H
#define LIGHTNING_PGFF_ENHANCED_OBS_MODEL_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <execution>

#include "common/eigen_types.h"
#include "common/point_def.h"
#include "core/pgff/predictive_lio.h"

namespace lightning {
namespace pgff {

/**
 * EnhancedObsModel - PGFF-Accelerated Observation Model
 * 
 * This class provides the key performance optimization:
 * Instead of processing ALL points in ObsModel, it:
 * 1. Predicts which points are "surprising" (informative)
 * 2. Only runs expensive nearest-neighbor + plane fitting on those
 * 3. Uses cached Jacobian rows for non-surprising points
 * 
 * Integration: Called from within the existing ObsModel lambda,
 * wrapping the expensive per-point operations.
 */
class EnhancedObsModel {
public:
    struct ProcessingResult {
        bool valid = false;
        float residual = 0;
        Vec4f plane_coef;
        Vec3f point_world;
        bool from_cache = false;
    };

    /**
     * Point processing function type
     * This matches the signature used in LaserMapping::ObsModel
     */
    using PointProcessor = std::function<ProcessingResult(int point_index)>;

    EnhancedObsModel(PredictiveLIO& pgff) : pgff_(pgff) {}

    /**
     * Process points with selective evaluation
     * 
     * @param num_points Total number of points
     * @param residuals Pre-computed residuals for surprise detection
     * @param full_processor Function to process a point fully (expensive)
     * @param results Output processing results
     * @return Number of points actually processed (vs cached)
     */
    int ProcessSelectively(
        int num_points,
        const std::vector<float>& residuals,
        PointProcessor full_processor,
        std::vector<ProcessingResult>& results) {
        
        results.resize(num_points);
        
        if (!pgff_.IsEnabled()) {
            // PGFF disabled - process all points
            std::vector<size_t> indices(num_points);
            std::iota(indices.begin(), indices.end(), 0);
            
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                [&](size_t i) {
                    results[i] = full_processor(i);
                });
            
            return num_points;
        }
        
        // Get surprising points from PGFF
        // Note: We pass empty cloud here as we're using residual-based selection
        CloudPtr dummy_cloud(new PointCloudType());
        dummy_cloud->resize(num_points);
        
        std::vector<int> surprising = pgff_.SelectPointsToProcess(dummy_cloud, residuals);
        
        // Process surprising points fully
        std::for_each(std::execution::par_unseq, surprising.begin(), surprising.end(),
            [&](int i) {
                results[i] = full_processor(i);
                results[i].from_cache = false;
            });
        
        // Mark non-surprising points as cached (will use predicted values)
        int cached_count = 0;
        for (int i = 0; i < num_points; i++) {
            if (!pgff_.ShouldProcessPoint(i)) {
                results[i].valid = true;  // Assume valid from prediction
                results[i].from_cache = true;
                results[i].residual = (i < residuals.size()) ? residuals[i] : 0;
                cached_count++;
            }
        }
        
        return num_points - cached_count;
    }

    /**
     * Build Jacobian with selective computation
     * 
     * For surprising points: Compute full Jacobian row
     * For expected points: Use cached/predicted Jacobian row
     */
    void BuildSelectiveJacobian(
        int num_effective_points,
        const std::vector<int>& point_indices,
        const std::vector<bool>& is_surprising,
        std::function<void(int, Eigen::Matrix<double, 1, 12>&)> compute_row,
        Eigen::MatrixXd& H_out,
        Eigen::VectorXd& residuals_out) {
        
        H_out.resize(num_effective_points, 12);
        residuals_out.resize(num_effective_points);
        
        // Try to get cached Jacobian for non-surprising points
        std::vector<int> non_surprising;
        for (size_t i = 0; i < point_indices.size(); i++) {
            if (!is_surprising[i]) {
                non_surprising.push_back(point_indices[i]);
            }
        }
        
        Eigen::MatrixXd H_cached;
        Eigen::VectorXd res_cached;
        bool have_cache = pgff_.GetCachedJacobianRows(non_surprising, H_cached, res_cached);
        
        // Build Jacobian
        int cache_idx = 0;
        std::vector<size_t> indices(num_effective_points);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
            [&](size_t i) {
                if (i < is_surprising.size() && is_surprising[i]) {
                    // Compute full row for surprising point
                    Eigen::Matrix<double, 1, 12> row;
                    compute_row(point_indices[i], row);
                    H_out.row(i) = row;
                } else if (have_cache && cache_idx < H_cached.rows()) {
                    // Use cached row
                    H_out.row(i) = H_cached.row(cache_idx);
                    cache_idx++;
                } else {
                    // Fallback: compute anyway
                    Eigen::Matrix<double, 1, 12> row;
                    compute_row(point_indices[i], row);
                    H_out.row(i) = row;
                }
            });
    }

private:
    PredictiveLIO& pgff_;
};

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_PGFF_ENHANCED_OBS_MODEL_H
