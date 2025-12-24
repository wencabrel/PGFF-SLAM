//
// Predictive Geometric Flow Fields (PGFF)
// Surprise Detector - Information-theoretic point selection
//

#ifndef LIGHTNING_PGFF_SURPRISE_DETECTOR_H
#define LIGHTNING_PGFF_SURPRISE_DETECTOR_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "common/eigen_types.h"
#include "common/point_def.h"
#include "core/pgff/flow_field.h"

namespace lightning {
namespace pgff {

/**
 * SurpriseDetector - Core PGFF Selection Mechanism
 * 
 * Implements the key insight from predictive coding:
 * Only prediction ERRORS carry information.
 * 
 * A point is "surprising" if:
 * 1. Its position differs significantly from prediction
 * 2. Its surface normal differs from expected
 * 3. It appears where we predicted empty space (or vice versa)
 * 
 * By only processing surprising points, we achieve:
 * - O(k) instead of O(n) where k << n
 * - Automatic attention to dynamic objects
 * - Implicit loop closure detection (familiar place = low surprise,
 *   unless the map is wrong = high surprise = potential loop)
 */
class SurpriseDetector {
public:
    struct Options {
        // Geometric surprise thresholds
        double position_surprise_threshold;
        double normal_surprise_threshold;
        double residual_surprise_threshold;
        
        // Adaptive thresholding
        bool use_adaptive_threshold;
        double surprise_percentile;
        int min_surprising_points;
        int max_surprising_points;
        
        // Information weighting
        double information_weight_position;
        double information_weight_normal;
        double information_weight_residual;
        
        Options() : position_surprise_threshold(0.1),
                    normal_surprise_threshold(0.3),
                    residual_surprise_threshold(0.15),
                    use_adaptive_threshold(true),
                    surprise_percentile(0.25),
                    min_surprising_points(100),
                    max_surprising_points(2000),
                    information_weight_position(1.0),
                    information_weight_normal(0.5),
                    information_weight_residual(2.0) {}
    };

    struct PointSurprise {
        int point_index;
        float surprise_score;      // Combined surprise metric
        float position_surprise;   // How far from predicted position
        float normal_surprise;     // How different from predicted normal
        float residual_surprise;   // Prediction vs actual residual
        bool is_surprising;        // Above threshold?
    };

    struct SurpriseStatistics {
        double mean_surprise = 0;
        double max_surprise = 0;
        double surprise_std = 0;
        int num_surprising = 0;
        int num_expected = 0;
        double adaptive_threshold = 0;
        double information_density = 0;  // bits per point
    };

    SurpriseDetector(Options options = Options()) : options_(options) {}

    /**
     * Compute surprise scores for all points
     * @param actual_points Actual observed points
     * @param predicted_corrs Predicted correspondences from flow field
     * @param actual_residuals Actual point-to-plane residuals
     * @param predicted_residuals Predicted residuals
     * @return Vector of surprise information per point
     */
    std::vector<PointSurprise> ComputeSurprise(
        const CloudPtr& actual_points,
        const std::vector<GeometricFlowField::PredictedCorrespondence>& predicted_corrs,
        const std::vector<float>& actual_residuals,
        const std::vector<float>& predicted_residuals);

    /**
     * Select which points to process based on surprise
     * @param surprises Per-point surprise information
     * @return Indices of points that should be processed
     */
    std::vector<int> SelectSurprisingPoints(
        const std::vector<PointSurprise>& surprises);

    /**
     * Fast surprise computation using only residuals
     * (when full prediction isn't available)
     */
    std::vector<bool> QuickSurpriseFromResiduals(
        const std::vector<float>& residuals,
        const std::vector<float>& predicted_residuals);

    /**
     * Compute information content of a point
     * Based on surprise and local geometry
     */
    float ComputeInformationContent(const PointSurprise& surprise) const;

    /**
     * Adaptive threshold computation
     * Adjusts based on scene statistics
     */
    double ComputeAdaptiveThreshold(const std::vector<float>& surprise_scores);

    const SurpriseStatistics& GetStatistics() const { return stats_; }
    const Options& GetOptions() const { return options_; }
    void SetOptions(const Options& options) { options_ = options; }

private:
    float ComputePositionSurprise(
        const Vec3f& actual,
        const Vec3f& predicted,
        float confidence) const;

    float ComputeNormalSurprise(
        const Vec3f& actual_normal,
        const Vec3f& predicted_normal) const;

    float ComputeResidualSurprise(
        float actual_residual,
        float predicted_residual) const;

    Options options_;
    SurpriseStatistics stats_;
};

// Inline implementations for performance-critical functions

inline float SurpriseDetector::ComputePositionSurprise(
    const Vec3f& actual,
    const Vec3f& predicted,
    float confidence) const {
    
    float distance = (actual - predicted).norm();
    // Scale by inverse confidence - low confidence predictions
    // should have higher tolerance
    float effective_threshold = options_.position_surprise_threshold / 
                                std::max(0.1f, confidence);
    return distance / effective_threshold;
}

inline float SurpriseDetector::ComputeNormalSurprise(
    const Vec3f& actual_normal,
    const Vec3f& predicted_normal) const {
    
    // Angle between normals
    float dot = std::clamp(actual_normal.dot(predicted_normal), -1.0f, 1.0f);
    float angle = std::acos(std::abs(dot));  // abs because normals can flip
    return angle / options_.normal_surprise_threshold;
}

inline float SurpriseDetector::ComputeResidualSurprise(
    float actual_residual,
    float predicted_residual) const {
    
    float diff = std::abs(actual_residual - predicted_residual);
    return diff / options_.residual_surprise_threshold;
}

inline float SurpriseDetector::ComputeInformationContent(
    const PointSurprise& surprise) const {
    
    // Information ∝ -log(probability of observation)
    // Surprising observations are less probable, hence more informative
    // Using sigmoid to bound the information content
    
    float s = surprise.surprise_score;
    float info = std::log1p(s * s);  // Quadratic scaling for high surprise
    
    return info;
}

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_PGFF_SURPRISE_DETECTOR_H
