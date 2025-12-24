#pragma once
// Learned Surprise Prior (Option 4) - Adaptive surprise threshold learning
// This system learns what "normal" surprise looks like for different geometry types
// and adapts thresholds accordingly for more intelligent point weighting

#include <Eigen/Eigen>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <numeric>

namespace lightning {
namespace pgff {

// Geometry classification based on local structure
enum class GeometryType : uint8_t {
    kUnknown = 0,
    kPlanar = 1,        // Walls, floors (low eigenvalue ratio)
    kLinear = 2,        // Edges, poles (medium eigenvalue ratio)
    kScattered = 3,     // Foliage, cluttered (high eigenvalue ratio)
    kCorner = 4,        // Corners, intersections
    kDynamic = 5        // Moving objects (high temporal variance)
};

// Per-geometry-type surprise statistics
struct SurpriseStats {
    double mean{0.0};
    double variance{0.0};
    double min{1e9};
    double max{0.0};
    int sample_count{0};
    
    std::deque<double> recent_samples;  // For online learning
    static constexpr int kMaxSamples = 500;
    
    void AddSample(double surprise) {
        recent_samples.push_back(surprise);
        if (recent_samples.size() > kMaxSamples) {
            recent_samples.pop_front();
        }
        
        // Online Welford's algorithm for mean/variance
        sample_count++;
        double delta = surprise - mean;
        mean += delta / sample_count;
        variance += delta * (surprise - mean);
        
        min = std::min(min, surprise);
        max = std::max(max, surprise);
    }
    
    double GetStdDev() const {
        if (sample_count < 2) return 1.0;
        return std::sqrt(variance / (sample_count - 1));
    }
    
    // Z-score: how many standard deviations from mean
    double GetZScore(double surprise) const {
        double stddev = GetStdDev();
        if (stddev < 1e-6) return 0.0;
        return (surprise - mean) / stddev;
    }
    
    // Adaptive threshold: mean + k*stddev
    double GetAdaptiveThreshold(double k = 1.5) const {
        return mean + k * GetStdDev();
    }
};

// Spatial region key for region-based learning
struct RegionKey {
    int64_t x, y, z;
    
    bool operator==(const RegionKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct RegionKeyHash {
    size_t operator()(const RegionKey& k) const {
        // FNV-1a hash
        size_t hash = 14695981039346656037ULL;
        hash = (hash ^ k.x) * 1099511628211ULL;
        hash = (hash ^ k.y) * 1099511628211ULL;
        hash = (hash ^ k.z) * 1099511628211ULL;
        return hash;
    }
};

// Per-region surprise learning
struct RegionPrior {
    SurpriseStats stats;
    GeometryType dominant_type{GeometryType::kUnknown};
    int visit_count{0};
    double last_surprise{0.0};
    
    // Temporal coherence: track surprise changes over time
    std::deque<double> temporal_surprises;
    static constexpr int kTemporalWindow = 20;
    
    void AddObservation(double surprise, GeometryType type) {
        stats.AddSample(surprise);
        last_surprise = surprise;
        visit_count++;
        
        temporal_surprises.push_back(surprise);
        if (temporal_surprises.size() > kTemporalWindow) {
            temporal_surprises.pop_front();
        }
        
        // Update dominant geometry type (simple mode tracking)
        dominant_type = type;  // Simplified: last seen type
    }
    
    // Get temporal variance (high = dynamic region)
    double GetTemporalVariance() const {
        if (temporal_surprises.size() < 3) return 0.0;
        
        double mean_t = std::accumulate(temporal_surprises.begin(), 
                                        temporal_surprises.end(), 0.0) / temporal_surprises.size();
        double var = 0.0;
        for (double s : temporal_surprises) {
            var += (s - mean_t) * (s - mean_t);
        }
        return var / temporal_surprises.size();
    }
    
    bool IsDynamic() const {
        return GetTemporalVariance() > 0.01;  // High temporal variance
    }
};

// Main Learned Surprise Prior system
class LearnedSurprisePrior {
public:
    struct Config {
        double region_size = 10.0;           // Size of spatial regions (meters)
        double learning_rate = 0.1;          // How fast to adapt
        double prior_weight = 0.3;           // Weight of prior vs observation
        double novelty_bonus = 1.5;          // Bonus for novel regions
        double dynamic_penalty = 0.5;        // Penalty for dynamic regions
        
        // Z-score thresholds for classification
        double high_surprise_z = 1.5;        // Above this = boost weight
        double low_surprise_z = -1.0;        // Below this = reduce weight
        
        Config() {}  // Explicit default constructor
    };
    
    explicit LearnedSurprisePrior(const Config& config = Config()) : config_(config) {
        // Initialize global priors for each geometry type
        InitializeGeometryPriors();
    }
    
    // Classify geometry based on eigenvalues (from point neighborhood)
    static GeometryType ClassifyGeometry(const Eigen::Vector3d& eigenvalues) {
        double e1 = eigenvalues(0);  // Largest
        double e2 = eigenvalues(1);
        double e3 = eigenvalues(2);  // Smallest
        
        double sum = e1 + e2 + e3 + 1e-6;
        double linearity = (e1 - e2) / sum;
        double planarity = (e2 - e3) / sum;
        double scattering = e3 / sum;
        
        if (planarity > 0.4) return GeometryType::kPlanar;
        if (linearity > 0.4) return GeometryType::kLinear;
        if (scattering > 0.3) return GeometryType::kScattered;
        if (planarity > 0.2 && linearity > 0.2) return GeometryType::kCorner;
        
        return GeometryType::kUnknown;
    }
    
    // Get adaptive weight based on learned priors
    double GetAdaptiveWeight(double raw_surprise, 
                             const Eigen::Vector3d& position,
                             GeometryType geo_type = GeometryType::kUnknown) {
        
        RegionKey key = GetRegionKey(position);
        
        // Get or create region prior
        auto& region = region_priors_[key];
        
        // Get geometry-type prior
        SurpriseStats& geo_prior = geometry_priors_[static_cast<int>(geo_type)];
        
        // Compute Z-score relative to both priors
        double z_region = region.stats.GetZScore(raw_surprise);
        double z_geo = geo_prior.GetZScore(raw_surprise);
        
        // Combine Z-scores (weighted average)
        double region_weight = std::min(1.0, region.visit_count / 50.0);  // Trust region more as visits increase
        double z_combined = region_weight * z_region + (1.0 - region_weight) * z_geo;
        
        // Convert Z-score to weight multiplier
        double weight = 1.0;
        
        if (z_combined > config_.high_surprise_z) {
            // High surprise relative to prior -> boost
            weight = 1.0 + 0.5 * (z_combined - config_.high_surprise_z);
            weight = std::min(3.0, weight);
        } else if (z_combined < config_.low_surprise_z) {
            // Low surprise relative to prior -> reduce
            weight = 1.0 + 0.3 * (z_combined - config_.low_surprise_z);
            weight = std::max(0.3, weight);
        }
        
        // Novelty bonus for rarely-visited regions
        if (region.visit_count < 5) {
            weight *= config_.novelty_bonus;
        }
        
        // Dynamic region penalty (unreliable)
        if (region.IsDynamic()) {
            weight *= config_.dynamic_penalty;
        }
        
        return weight;
    }
    
    // Update priors with new observation
    void UpdatePrior(double surprise, 
                     const Eigen::Vector3d& position,
                     GeometryType geo_type) {
        
        RegionKey key = GetRegionKey(position);
        
        // Update region prior
        region_priors_[key].AddObservation(surprise, geo_type);
        
        // Update geometry-type prior
        geometry_priors_[static_cast<int>(geo_type)].AddSample(surprise);
        
        total_observations_++;
    }
    
    // Batch update for efficiency
    void UpdateBatch(const std::vector<double>& surprises,
                     const std::vector<Eigen::Vector3d>& positions,
                     const std::vector<GeometryType>& types) {
        
        for (size_t i = 0; i < surprises.size(); ++i) {
            UpdatePrior(surprises[i], positions[i], types[i]);
        }
    }
    
    // Get expected surprise for a position (prediction)
    double PredictSurprise(const Eigen::Vector3d& position,
                           GeometryType geo_type) const {
        
        RegionKey key = GetRegionKey(position);
        
        auto it = region_priors_.find(key);
        if (it != region_priors_.end() && it->second.visit_count > 10) {
            // Have enough regional data
            return it->second.stats.mean;
        }
        
        // Fall back to geometry-type prior
        return geometry_priors_[static_cast<int>(geo_type)].mean;
    }
    
    // Get learning statistics
    struct LearningStats {
        int total_observations;
        int num_regions;
        int num_dynamic_regions;
        double global_mean_surprise;
        double global_variance;
        std::array<double, 6> geometry_means;  // Per-type means
    };
    
    LearningStats GetStats() const {
        LearningStats stats;
        stats.total_observations = total_observations_;
        stats.num_regions = region_priors_.size();
        stats.num_dynamic_regions = 0;
        
        double sum = 0.0, sum_sq = 0.0;
        int count = 0;
        
        for (const auto& [key, region] : region_priors_) {
            if (region.IsDynamic()) stats.num_dynamic_regions++;
            sum += region.stats.mean * region.visit_count;
            count += region.visit_count;
        }
        
        stats.global_mean_surprise = count > 0 ? sum / count : 0.0;
        
        for (int i = 0; i < 6; ++i) {
            stats.geometry_means[i] = geometry_priors_[i].mean;
        }
        
        return stats;
    }
    
    // Periodic logging
    void LogLearningProgress(int frame_id) const {
        if (frame_id % 50 != 0) return;  // Log every 50 frames
        
        auto stats = GetStats();
        printf("[SurprisePrior] Frame %d: obs=%d regions=%d dynamic=%d global_mean=%.4f\n",
               frame_id, stats.total_observations, stats.num_regions,
               stats.num_dynamic_regions, stats.global_mean_surprise);
        printf("  Geometry means: planar=%.4f linear=%.4f scatter=%.4f corner=%.4f\n",
               stats.geometry_means[1], stats.geometry_means[2],
               stats.geometry_means[3], stats.geometry_means[4]);
    }

private:
    Config config_;
    
    // Per-geometry-type priors
    std::array<SurpriseStats, 6> geometry_priors_;
    
    // Spatial region priors
    std::unordered_map<RegionKey, RegionPrior, RegionKeyHash> region_priors_;
    
    int total_observations_{0};
    
    RegionKey GetRegionKey(const Eigen::Vector3d& pos) const {
        return {
            static_cast<int64_t>(std::floor(pos.x() / config_.region_size)),
            static_cast<int64_t>(std::floor(pos.y() / config_.region_size)),
            static_cast<int64_t>(std::floor(pos.z() / config_.region_size))
        };
    }
    
    void InitializeGeometryPriors() {
        // Initialize with reasonable default priors based on typical SLAM data
        // These will be refined as data comes in
        
        // Unknown
        geometry_priors_[0].mean = 0.05;
        geometry_priors_[0].variance = 0.01;
        geometry_priors_[0].sample_count = 100;  // Pseudo-count
        
        // Planar - typically low surprise
        geometry_priors_[1].mean = 0.03;
        geometry_priors_[1].variance = 0.005;
        geometry_priors_[1].sample_count = 100;
        
        // Linear - medium surprise
        geometry_priors_[2].mean = 0.04;
        geometry_priors_[2].variance = 0.008;
        geometry_priors_[2].sample_count = 100;
        
        // Scattered - high surprise (noisy)
        geometry_priors_[3].mean = 0.08;
        geometry_priors_[3].variance = 0.02;
        geometry_priors_[3].sample_count = 100;
        
        // Corner - medium-high surprise
        geometry_priors_[4].mean = 0.05;
        geometry_priors_[4].variance = 0.01;
        geometry_priors_[4].sample_count = 100;
        
        // Dynamic - high surprise, high variance
        geometry_priors_[5].mean = 0.10;
        geometry_priors_[5].variance = 0.05;
        geometry_priors_[5].sample_count = 100;
    }
};

}  // namespace pgff
}  // namespace lightning
