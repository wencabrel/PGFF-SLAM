#include "ui/ui_car.h"
#include <GL/gl.h>

namespace lightning::ui {

// Robot/Vehicle outline vertices - a recognizable car/robot shape from top view
std::vector<Vec3f> UiCar::car_vertices_ = {
    // clang-format off
    // Main body outline (rectangle with pointed front)
    { 2.5,  0.0, 0.0},   // Front point (nose)
    { 1.5,  1.0, 0.0},   // Front-right
    {-1.5,  1.0, 0.0},   // Back-right  
    {-2.0,  0.5, 0.0},   // Back-right corner
    {-2.0, -0.5, 0.0},   // Back-left corner
    {-1.5, -1.0, 0.0},   // Back-left
    { 1.5, -1.0, 0.0},   // Front-left
    { 2.5,  0.0, 0.0},   // Back to front point
    // clang-format on
};

void UiCar::SetPose(const SE3& pose) {
    pts_.clear();
    for (auto& p : car_vertices_) {
        pts_.emplace_back(p);
    }

    // 转换到世界系
    auto pose_f = pose.cast<float>();
    for (auto& pt : pts_) {
        pt = pose_f * pt;
    }
}

void UiCar::Render() {
    // Draw filled polygon for body
    glLineWidth(3.0);
    
    // Draw car outline
    glBegin(GL_LINE_LOOP);
    glColor4f(color_[0], color_[1], color_[2], 0.9f);
    for (const auto& pt : pts_) {
        glVertex3f(pt[0], pt[1], pt[2]);
    }
    glEnd();
    
    // Draw filled body with slight transparency
    glBegin(GL_POLYGON);
    glColor4f(color_[0], color_[1], color_[2], 0.4f);
    for (const auto& pt : pts_) {
        glVertex3f(pt[0], pt[1], pt[2]);
    }
    glEnd();
    
    // Draw forward direction indicator (arrow from center to front)
    glLineWidth(4.0);
    glBegin(GL_LINES);
    glColor4f(1.0f, 0.8f, 0.2f, 1.0f);  // Yellow/gold for direction
    // Center to front
    Vec3f center = (pts_[0] + pts_[3] + pts_[4]) / 3.0f;  // Approximate center
    glVertex3f(center[0], center[1], center[2] + 0.5f);
    glVertex3f(pts_[0][0], pts_[0][1], pts_[0][2] + 0.5f);  // Front point
    glEnd();
    
    // Draw coordinate frame at robot center (smaller)
    glLineWidth(2.0);
    auto pose_f = Sophus::SE3f();  // Use identity for relative frame
    Vec3f origin = center;
    origin[2] += 0.3f;  // Slightly above
    
    float axis_len = 1.5f;
    glBegin(GL_LINES);
    // X axis - Red
    glColor3f(1.0, 0.3, 0.3);
    glVertex3f(origin[0], origin[1], origin[2]);
    glVertex3f(origin[0] + axis_len, origin[1], origin[2]);
    // Y axis - Green  
    glColor3f(0.3, 1.0, 0.3);
    glVertex3f(origin[0], origin[1], origin[2]);
    glVertex3f(origin[0], origin[1] + axis_len, origin[2]);
    // Z axis - Blue
    glColor3f(0.3, 0.3, 1.0);
    glVertex3f(origin[0], origin[1], origin[2]);
    glVertex3f(origin[0], origin[1], origin[2] + axis_len);
    glEnd();
}

}  // namespace lightning::ui
