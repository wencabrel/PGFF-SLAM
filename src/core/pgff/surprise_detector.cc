//
// Predictive Geometric Flow Fields (PGFF)
// Surprise Detector Implementation
//

#include "core/pgff/surprise_detector.h"
#include <algorithm>
#include <cmath>
#include <execution>

namespace lightning {
namespace pgff {

std::vector<SurpriseDetector::PointSurprise> SurpriseDetector::ComputeSurprise(
    const CloudPtr& actual_points,
    const std::vector<GeometricFlowField::PredictedCorrespondence>& predicted_corrs,
    const std::vector<float>& actual_residuals,
    const std::vector<float>& predicted_residuals) {
    
    int num_points = actual_points->size();
    std::vector<PointSurprise> surprises(num_points);
    
    // Compute surprise for each point
    std::vector<size_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
        [&](size_t i) {
            surprises[i].point_index = i;
            
            const auto& pt = actual_points->points[i];
            Vec3f actual_pos(pt.x, pt.y, pt.z);
            
            // If we have prediction for this point
            if (i < predicted_corrs.size() && predicted_corrs[i].valid) {
                const auto& pred = predicted_corrs[i];
                
                // Position surprise
                surprises[i].position_surprise = ComputePositionSurprise(
                    actual_pos, pred.predicted_point, pred.confidence);
                
                // Normal surprise (if available)
                // For now, estimate from plane coefficient
                surprises[i].normal_surprise = 0;  // Will be computed when available
                
            } else {
                // No prediction = maximum surprise (novel observation)
                surprises[i].position_surprise = 2.0f;
                surprises[i].normal_surprise = 1.0f;
            }
            
            // Residual surprise
            if (i < actual_residuals.size() && i < predicted_residuals.size()) {
                surprises[i].residual_surprise = ComputeResidualSurprise(
                    actual_residuals[i], predicted_residuals[i]);
            } else if (i < actual_residuals.size()) {
                // No prediction, use absolute residual as surprise
                surprises[i].residual_surprise = 
                    std::abs(actual_residuals[i]) / options_.residual_surprise_threshold;
            }
            
            // Combined surprise score (weighted sum)
            surprises[i].surprise_score = 
                options_.information_weight_position * surprises[i].position_surprise +
                options_.information_weight_normal * surprises[i].normal_surprise +
                options_.information_weight_residual * surprises[i].residual_surprise;
        });
    
    // Compute statistics
    double sum = 0, sum_sq = 0;
    float max_s = 0;
    for (const auto& s : surprises) {
        sum += s.surprise_score;
        sum_sq += s.surprise_score * s.surprise_score;
        max_s = std::max(max_s, s.surprise_score);
    }
    
    stats_.mean_surprise = sum / num_points;
    stats_.surprise_std = std::sqrt(sum_sq / num_points - 
                                     stats_.mean_surprise * stats_.mean_surprise);
    stats_.max_surprise = max_s;
    
    // Determine threshold and mark surprising points
    double threshold = options_.use_adaptive_threshold ?
        ComputeAdaptiveThreshold(
            [&]() {
                std::vector<float> scores(num_points);
                for (int i = 0; i < num_points; i++) {
                    scores[i] = surprises[i].surprise_score;
                }
                return scores;
            }()) :
        1.0;  // Default threshold
    
    stats_.adaptive_threshold = threshold;
    
    int num_surprising = 0;
    for (auto& s : surprises) {
        s.is_surprising = s.surprise_score > threshold;
        if (s.is_surprising) num_surprising++;
    }
    
    stats_.num_surprising = num_surprising;
    stats_.num_expected = num_points - num_surprising;
    
    return surprises;
}

std::vector<int> SurpriseDetector::SelectSurprisingPoints(
    const std::vector<PointSurprise>& surprises) {
    
    std::vector<int> selected;
    selected.reserve(options_.max_surprising_points);
    
    // First pass: collect all surprising points
    std::vector<std::pair<float, int>> scored_points;
    for (const auto& s : surprises) {
        if (s.is_surprising) {
            scored_points.emplace_back(s.surprise_score, s.point_index);
        }
    }
    
    // If we have too few, add highest-scoring non-surprising points
    if (scored_points.size() < options_.min_surprising_points) {
        for (const auto& s : surprises) {
            if (!s.is_surprising) {
                scored_points.emplace_back(s.surprise_score, s.point_index);
            }
        }
    }
    
    // Sort by surprise score (descending)
    std::sort(scored_points.begin(), scored_points.end(),
              std::greater<std::pair<float, int>>());
    
    // Select top points up to max
    int num_select = std::min(static_cast<int>(scored_points.size()),
                              options_.max_surprising_points);
    num_select = std::max(num_select, options_.min_surprising_points);
    num_select = std::min(num_select, static_cast<int>(scored_points.size()));
    
    for (int i = 0; i < num_select; i++) {
        selected.push_back(scored_points[i].second);
    }
    
    // Sort by index for cache-friendly access
    std::sort(selected.begin(), selected.end());
    
    return selected;
}

std::vector<bool> SurpriseDetector::QuickSurpriseFromResiduals(
    const std::vector<float>& residuals,
    const std::vector<float>& predicted_residuals) {
    
    int n = residuals.size();
    std::vector<bool> is_surprising(n, false);
    
    if (predicted_residuals.empty()) {
        // No predictions - use absolute residual threshold
        // Take top percentile by residual magnitude
        std::vector<std::pair<float, int>> abs_residuals(n);
        for (int i = 0; i < n; i++) {
            abs_residuals[i] = {std::abs(residuals[i]), i};
        }
        
        std::nth_element(abs_residuals.begin(),
                        abs_residuals.begin() + n * options_.surprise_percentile,
                        abs_residuals.end(),
                        std::greater<std::pair<float, int>>());
        
        int cutoff = std::max(options_.min_surprising_points,
                             static_cast<int>(n * options_.surprise_percentile));
        cutoff = std::min(cutoff, options_.max_surprising_points);
        
        for (int i = 0; i < cutoff && i < n; i++) {
            is_surprising[abs_residuals[i].second] = true;
        }
        
    } else {
        // Compare against predictions
        std::vector<std::pair<float, int>> surprise_scores(n);
        for (int i = 0; i < n; i++) {
            float pred = (i < predicted_residuals.size()) ? predicted_residuals[i] : 0;
            float surprise = std::abs(residuals[i] - pred);
            surprise_scores[i] = {surprise, i};
        }
        
        // Adaptive threshold
        std::vector<float> scores(n);
        for (int i = 0; i < n; i++) scores[i] = surprise_scores[i].first;
        double threshold = ComputeAdaptiveThreshold(scores);
        
        for (const auto& sp : surprise_scores) {
            if (sp.first > threshold) {
                is_surprising[sp.second] = true;
            }
        }
    }
    
    return is_surprising;
}

double SurpriseDetector::ComputeAdaptiveThreshold(
    const std::vector<float>& surprise_scores) {
    
    if (surprise_scores.empty()) return 1.0;
    
    // Find the threshold that selects the top percentile
    std::vector<float> sorted_scores = surprise_scores;
    
    int target_idx = static_cast<int>(
        sorted_scores.size() * (1.0 - options_.surprise_percentile));
    target_idx = std::max(0, std::min(target_idx, 
                          static_cast<int>(sorted_scores.size()) - 1));
    
    std::nth_element(sorted_scores.begin(),
                    sorted_scores.begin() + target_idx,
                    sorted_scores.end());
    
    double threshold = sorted_scores[target_idx];
    
    // Ensure we get at least min_surprising_points
    if (sorted_scores.size() > options_.min_surprising_points) {
        int min_idx = sorted_scores.size() - options_.min_surprising_points;
        std::nth_element(sorted_scores.begin(),
                        sorted_scores.begin() + min_idx,
                        sorted_scores.end());
        threshold = std::min(threshold, static_cast<double>(sorted_scores[min_idx]));
    }
    
    return threshold;
}

}  // namespace pgff
}  // namespace lightning
