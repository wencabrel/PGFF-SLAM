//
// Information Frontier Tracking for Predictive PGFF
// Part of the Groundbreaking PGFF Enhancement (Option 3)
//

#ifndef LIGHTNING_INFORMATION_FRONTIER_H
#define LIGHTNING_INFORMATION_FRONTIER_H

#include "common/eigen_types.h"
#include <deque>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace lightning {
namespace pgff {

/**
 * Information Frontier Cell
 * Represents a region of the map with information metrics
 */
struct FrontierCell {
    Vec3d center;                    // Cell center in world frame
    double information_gain = 0.0;   // Predicted information gain
    double uncertainty = 1.0;        // Current uncertainty (entropy)
    double observation_count = 0;    // How many times observed
    int last_observed_frame = -1;    // Last frame when observed
    double surprise_history = 0.0;   // Accumulated surprise when visited
    
    // Prediction tracking
    double predicted_surprise = 0.0;  // What we predicted surprise would be
    double actual_surprise = 0.0;     // What surprise actually was
    double prediction_error = 0.0;    // |predicted - actual|
    
    bool is_frontier = false;        // Is this an exploration frontier?
    
    FrontierCell() = default;
    FrontierCell(const Vec3d& c) : center(c) {}
};

/**
 * Information Frontier Manager
 * Tracks map regions, predicts information gain, and validates predictions
 */
class InformationFrontier {
   public:
    struct Config {
        double cell_size = 5.0;           // Grid cell size in meters
        int prediction_horizon = 10;       // Frames ahead to predict
        double decay_rate = 0.95;          // Information decay over time
        double frontier_threshold = 0.7;   // Uncertainty threshold for frontier
        int history_size = 100;            // Size of prediction history
        double learning_rate = 0.1;        // How fast to update predictions
        
        Config() = default;
    };
    
    InformationFrontier() : config_() {}
    explicit InformationFrontier(const Config& config) : config_(config) {}
    
    /**
     * Update frontier with new observation
     * @param position Current robot position
     * @param frame_id Current frame ID
     * @param surprise Measured PGFF surprise for this frame
     */
    void Update(const Vec3d& position, int frame_id, double surprise) {
        // Get cell for current position
        auto key = PositionToKey(position);
        auto& cell = cells_[key];
        
        if (cell.observation_count == 0) {
            cell.center = KeyToPosition(key);
        }
        
        // Update prediction accuracy if we had a prediction
        if (cell.predicted_surprise > 0) {
            cell.actual_surprise = surprise;
            cell.prediction_error = std::abs(cell.predicted_surprise - surprise);
            
            // Track overall prediction accuracy
            prediction_errors_.push_back(cell.prediction_error);
            if (prediction_errors_.size() > config_.history_size) {
                prediction_errors_.pop_front();
            }
            
            // Update our prediction model with this observation
            UpdatePredictionModel(cell, surprise);
        }
        
        // Update cell state
        cell.observation_count++;
        cell.last_observed_frame = frame_id;
        cell.surprise_history = config_.decay_rate * cell.surprise_history + surprise;
        
        // Reduce uncertainty as we observe
        cell.uncertainty *= (1.0 - 0.1 * std::min(1.0, surprise));
        
        // Update frontier status
        cell.is_frontier = (cell.uncertainty > config_.frontier_threshold);
        
        // Update neighboring cells
        UpdateNeighborFrontiers(key, frame_id);
        
        // Compute information gain predictions for nearby unvisited cells
        PredictNearbyInformationGain(position, frame_id);
        
        current_frame_ = frame_id;
        current_position_ = position;
    }
    
    /**
     * Get predicted information gain for a position
     */
    double GetPredictedInformationGain(const Vec3d& position) const {
        auto key = PositionToKey(position);
        auto it = cells_.find(key);
        if (it != cells_.end()) {
            return it->second.information_gain;
        }
        return 1.0;  // Unknown areas have high predicted information
    }
    
    /**
     * Get predicted surprise for a position
     */
    double GetPredictedSurprise(const Vec3d& position) const {
        auto key = PositionToKey(position);
        auto it = cells_.find(key);
        if (it != cells_.end()) {
            return it->second.predicted_surprise;
        }
        // For unvisited areas, predict based on distance and neighborhood
        return PredictSurpriseForUnvisited(position);
    }
    
    /**
     * Get all frontier cells (high uncertainty, exploration targets)
     */
    std::vector<FrontierCell> GetFrontierCells() const {
        std::vector<FrontierCell> frontiers;
        for (const auto& [key, cell] : cells_) {
            if (cell.is_frontier) {
                frontiers.push_back(cell);
            }
        }
        return frontiers;
    }
    
    /**
     * Get the best exploration direction from current position
     * Returns direction vector towards highest information gain
     */
    Vec3d GetBestExplorationDirection(const Vec3d& from_position) const {
        Vec3d best_direction = Vec3d::Zero();
        double best_score = -1.0;
        
        // Check nearby cells
        for (int dx = -3; dx <= 3; dx++) {
            for (int dy = -3; dy <= 3; dy++) {
                if (dx == 0 && dy == 0) continue;
                
                Vec3d candidate = from_position + Vec3d(dx * config_.cell_size, 
                                                        dy * config_.cell_size, 0);
                double score = GetPredictedInformationGain(candidate);
                
                // Favor unexplored areas
                auto key = PositionToKey(candidate);
                if (cells_.find(key) == cells_.end()) {
                    score *= 1.5;  // Bonus for unexplored
                }
                
                if (score > best_score) {
                    best_score = score;
                    best_direction = (candidate - from_position).normalized();
                }
            }
        }
        
        return best_direction;
    }
    
    /**
     * Get prediction accuracy statistics
     */
    struct PredictionStats {
        double mean_error = 0.0;
        double max_error = 0.0;
        double accuracy = 0.0;  // 1.0 - normalized error
        int num_predictions = 0;
    };
    
    PredictionStats GetPredictionStats() const {
        PredictionStats stats;
        if (prediction_errors_.empty()) return stats;
        
        stats.num_predictions = prediction_errors_.size();
        double sum = 0.0;
        stats.max_error = 0.0;
        
        for (double err : prediction_errors_) {
            sum += err;
            stats.max_error = std::max(stats.max_error, err);
        }
        
        stats.mean_error = sum / stats.num_predictions;
        stats.accuracy = std::max(0.0, 1.0 - stats.mean_error);
        
        return stats;
    }
    
    /**
     * Get total number of tracked cells
     */
    int GetCellCount() const { return cells_.size(); }
    
    /**
     * Get number of frontier cells
     */
    int GetFrontierCount() const {
        int count = 0;
        for (const auto& [key, cell] : cells_) {
            if (cell.is_frontier) count++;
        }
        return count;
    }
    
   private:
    using CellKey = std::tuple<int, int, int>;
    
    struct KeyHash {
        size_t operator()(const CellKey& k) const {
            return std::hash<int>()(std::get<0>(k)) ^ 
                   (std::hash<int>()(std::get<1>(k)) << 1) ^
                   (std::hash<int>()(std::get<2>(k)) << 2);
        }
    };
    
    CellKey PositionToKey(const Vec3d& pos) const {
        return {static_cast<int>(std::floor(pos.x() / config_.cell_size)),
                static_cast<int>(std::floor(pos.y() / config_.cell_size)),
                static_cast<int>(std::floor(pos.z() / config_.cell_size))};
    }
    
    Vec3d KeyToPosition(const CellKey& key) const {
        return Vec3d((std::get<0>(key) + 0.5) * config_.cell_size,
                     (std::get<1>(key) + 0.5) * config_.cell_size,
                     (std::get<2>(key) + 0.5) * config_.cell_size);
    }
    
    void UpdateNeighborFrontiers(const CellKey& center_key, int frame_id) {
        auto [cx, cy, cz] = center_key;
        
        // Update 6-connected neighbors
        std::vector<CellKey> neighbors = {
            {cx-1, cy, cz}, {cx+1, cy, cz},
            {cx, cy-1, cz}, {cx, cy+1, cz},
            {cx, cy, cz-1}, {cx, cy, cz+1}
        };
        
        for (const auto& nkey : neighbors) {
            auto it = cells_.find(nkey);
            if (it != cells_.end()) {
                // Neighbor exists - update its frontier status based on age
                auto& ncell = it->second;
                int age = frame_id - ncell.last_observed_frame;
                
                // Uncertainty grows with age
                ncell.uncertainty = std::min(1.0, ncell.uncertainty + 0.01 * age);
                ncell.is_frontier = (ncell.uncertainty > config_.frontier_threshold);
            }
        }
    }
    
    void PredictNearbyInformationGain(const Vec3d& position, int frame_id) {
        // Predict information gain for cells within prediction horizon
        for (int dx = -config_.prediction_horizon; dx <= config_.prediction_horizon; dx++) {
            for (int dy = -config_.prediction_horizon; dy <= config_.prediction_horizon; dy++) {
                Vec3d target = position + Vec3d(dx * config_.cell_size, 
                                                dy * config_.cell_size, 0);
                auto key = PositionToKey(target);
                
                auto it = cells_.find(key);
                if (it == cells_.end()) {
                    // New cell - predict based on neighbors
                    FrontierCell new_cell(KeyToPosition(key));
                    new_cell.predicted_surprise = PredictSurpriseFromNeighbors(key);
                    new_cell.information_gain = ComputeInformationGain(new_cell, frame_id);
                    new_cell.uncertainty = 1.0;  // Full uncertainty for unvisited
                    new_cell.is_frontier = true;
                    cells_[key] = new_cell;
                } else {
                    // Existing cell - update prediction
                    auto& cell = it->second;
                    cell.predicted_surprise = PredictSurpriseFromHistory(cell);
                    cell.information_gain = ComputeInformationGain(cell, frame_id);
                }
            }
        }
    }
    
    double PredictSurpriseFromNeighbors(const CellKey& key) const {
        auto [cx, cy, cz] = key;
        double sum = 0.0;
        int count = 0;
        
        // Average surprise of observed neighbors
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;
                
                auto nkey = CellKey{cx + dx, cy + dy, cz};
                auto it = cells_.find(nkey);
                if (it != cells_.end() && it->second.observation_count > 0) {
                    sum += it->second.surprise_history / it->second.observation_count;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            return sum / count;
        }
        return 0.05;  // Default surprise estimate for unknown areas
    }
    
    double PredictSurpriseFromHistory(const FrontierCell& cell) const {
        if (cell.observation_count == 0) {
            return 0.05;  // Default for unvisited
        }
        
        // Predict based on average historical surprise with decay
        double avg_surprise = cell.surprise_history / cell.observation_count;
        
        // Areas with high past surprise likely have high future surprise
        // But uncertainty also increases with time since last observation
        int age = current_frame_ - cell.last_observed_frame;
        double age_factor = 1.0 + 0.01 * age;  // Uncertainty grows with age
        
        return avg_surprise * age_factor;
    }
    
    double PredictSurpriseForUnvisited(const Vec3d& position) const {
        auto key = PositionToKey(position);
        return PredictSurpriseFromNeighbors(key);
    }
    
    double ComputeInformationGain(const FrontierCell& cell, int frame_id) const {
        // Information gain = uncertainty * expected surprise * accessibility
        double uncertainty_factor = cell.uncertainty;
        double surprise_factor = cell.predicted_surprise;
        
        // Decay information gain with age (old predictions are less reliable)
        int age = frame_id - cell.last_observed_frame;
        double freshness = std::exp(-0.1 * std::max(0, age));
        
        // Distance from current position affects accessibility
        double distance = (cell.center - current_position_).norm();
        double accessibility = 1.0 / (1.0 + 0.1 * distance);
        
        return uncertainty_factor * (0.5 + surprise_factor) * freshness * accessibility;
    }
    
    void UpdatePredictionModel(FrontierCell& cell, double actual_surprise) {
        // Simple exponential moving average update
        // The more we observe, the better our predictions should be
        double alpha = config_.learning_rate;
        
        // Adjust future predictions based on error
        if (cell.prediction_error > 0.1) {
            // Large error - need to adjust our model
            // If we under-predicted, increase future predictions
            // If we over-predicted, decrease future predictions
            double adjustment = alpha * (actual_surprise - cell.predicted_surprise);
            cell.surprise_history += adjustment;
        }
    }
    
    Config config_;
    std::unordered_map<CellKey, FrontierCell, KeyHash> cells_;
    std::deque<double> prediction_errors_;
    
    int current_frame_ = 0;
    Vec3d current_position_ = Vec3d::Zero();
};

}  // namespace pgff
}  // namespace lightning

#endif  // LIGHTNING_INFORMATION_FRONTIER_H
