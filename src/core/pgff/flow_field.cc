//
// Predictive Geometric Flow Fields (PGFF)
// Flow Field Implementation
//

#include "core/pgff/flow_field.h"
#include <cmath>

namespace lightning {
namespace pgff {

Vec3f GeometricFlowField::ComputePointFlow(
    const Vec3f& point_sensor_frame,
    const Vec3f& surface_normal,
    const SE3& relative_motion) const {
    
    // Key geometric insight:
    // As the sensor moves, a point on a surface appears to move in the opposite
    // direction, but constrained to the surface tangent plane
    
    // Get translation and rotation components
    Vec3f translation = relative_motion.translation().cast<float>();
    Mat3f rotation = relative_motion.so3().matrix().cast<float>();
    
    // Point motion in sensor frame due to ego-motion
    Vec3f ego_induced_motion = -(rotation * point_sensor_frame + translation - point_sensor_frame);
    
    // Project onto tangent plane (perpendicular to normal)
    // This accounts for surface continuity constraint
    Vec3f normal_component = surface_normal * surface_normal.dot(ego_induced_motion);
    Vec3f tangent_flow = ego_induced_motion - normal_component * options_.surface_coherence_weight;
    
    return tangent_flow;
}

Vec3f GeometricFlowField::PredictPointInNextFrame(
    const Vec3f& point_world,
    const Vec3f& normal_world,
    const SE3& current_pose,
    const SE3& predicted_pose) const {
    
    // Transform point to current sensor frame
    SE3 T_sensor_world_current = current_pose.inverse();
    Vec3f point_current_sensor = (T_sensor_world_current * point_world.cast<double>()).cast<float>();
    
    // Transform to predicted sensor frame
    SE3 T_sensor_world_predicted = predicted_pose.inverse();
    Vec3f point_predicted_sensor = (T_sensor_world_predicted * point_world.cast<double>()).cast<float>();
    
    // The predicted observation is simply the world point in the new sensor frame
    // But we can refine this using surface flow for occluded/edge points
    
    return point_predicted_sensor;
}

float GeometricFlowField::EstimatePredictionConfidence(
    const Vec3f& point,
    const Vec3f& normal,
    float local_density,
    const SE3& relative_motion) const {
    
    float confidence = 1.0f;
    
    // Factor 1: Surface regularity (normal stability indicates planar region)
    // Planar surfaces are highly predictable
    float normal_magnitude = normal.norm();
    if (normal_magnitude > 0.9f) {
        confidence *= 0.95f;  // Good normal = high confidence
    } else {
        confidence *= 0.5f + 0.5f * normal_magnitude;
    }
    
    // Factor 2: Motion magnitude (smaller motion = higher confidence)
    float trans_magnitude = relative_motion.translation().norm();
    float rot_magnitude = relative_motion.so3().log().norm();
    
    float motion_factor = std::exp(-trans_magnitude * 2.0f) * std::exp(-rot_magnitude * 5.0f);
    confidence *= (0.3f + 0.7f * motion_factor);
    
    // Factor 3: Point density (denser regions are more stable)
    float density_factor = std::min(1.0f, local_density / 10.0f);
    confidence *= (0.5f + 0.5f * density_factor);
    
    // Factor 4: Distance from sensor (closer = more reliable)
    float distance = point.norm();
    float distance_factor = std::exp(-distance * 0.05f);
    confidence *= (0.6f + 0.4f * distance_factor);
    
    return std::clamp(confidence, 0.0f, 1.0f);
}

}  // namespace pgff
}  // namespace lightning
