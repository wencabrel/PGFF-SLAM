#include "ui/ui_cloud.h"
#include "common/options.h"

#include <GL/gl.h>
#include <numeric>

namespace lightning::ui {

std::vector<Vec4f> UiCloud::intensity_color_table_pcl_;

UiCloud::UiCloud(CloudPtr cloud) { SetCloud(cloud, SE3()); }

void UiCloud::SetCustomColor(Vec4f custom_color) { custom_color_ = custom_color; }

// 把输入的点云映射为opengl可以渲染的点云
void UiCloud::SetCloud(CloudPtr cloud, const SE3& pose) {
    if (intensity_color_table_pcl_.empty()) {
        BuildIntensityTable();
    }

    // assert(cloud != nullptr && cloud->empty() == false);
    xyz_data_.resize(cloud->size());
    color_data_pcl_.resize(cloud->size());
    color_data_intensity_.resize(cloud->size());
    color_data_height_.resize(cloud->size());
    color_data_gray_.resize(cloud->size());
    color_data_uncertainty_.resize(cloud->size());

    std::vector<int> idx(cloud->size());
    std::iota(idx.begin(), idx.end(), 0);  // 使用从0开始递增的整数填充idx

    SE3f pose_l = (pose).cast<float>();

    // 遍历所有点
    for (auto iter = idx.begin(); iter != idx.end(); iter++) {
        const int& id = *iter;
        const auto& pt = cloud->points[id];
        // 计算点的世界坐标
        auto pt_world = pose_l * cloud->points[id].getVector3fMap();
        xyz_data_[id] = Vec3f(pt_world.x(), pt_world.y(), pt_world.z());
        // 把intensity映射为颜色
        color_data_pcl_[id] = IntensityToRgbPCL(pt.intensity);
        color_data_gray_[id] = Vec4f(0.6, 0.65, 0.7, 1.0);  // Light slate gray for better visibility
        // Default: green (confident) for uncertainty color
        color_data_uncertainty_[id] = Vec4f(0.2, 0.8, 0.2, 1.0);  // Green = confident
        // 根据高度映射颜色
        color_data_height_[id] = IntensityToRgbPCL(pt.z * 10);
        color_data_intensity_[id] =
            Vec4f(pt.intensity / 255.0 * 3.0, pt.intensity / 255.0 * 3.0, pt.intensity / 255.0 * 3.0, 1.0);
    }
}

// Set cloud with uncertainty coloring: 0=confident (green), 1=uncertain (red)
void UiCloud::SetCloudWithUncertainty(CloudPtr cloud, const SE3& pose, double uncertainty) {
    if (intensity_color_table_pcl_.empty()) {
        BuildIntensityTable();
    }

    xyz_data_.resize(cloud->size());
    color_data_pcl_.resize(cloud->size());
    color_data_intensity_.resize(cloud->size());
    color_data_height_.resize(cloud->size());
    color_data_gray_.resize(cloud->size());
    color_data_uncertainty_.resize(cloud->size());

    std::vector<int> idx(cloud->size());
    std::iota(idx.begin(), idx.end(), 0);

    SE3f pose_l = (pose).cast<float>();
    
    // Clamp uncertainty to [0, 1]
    float u = std::max(0.0, std::min(1.0, uncertainty));
    
    // Color gradient: green (0) -> yellow (0.5) -> red (1)
    Vec4f uncertainty_color;
    if (u < 0.5f) {
        // Green to yellow
        float t = u * 2.0f;
        uncertainty_color = Vec4f(t, 0.8f, 0.2f * (1.0f - t), 1.0f);
    } else {
        // Yellow to red
        float t = (u - 0.5f) * 2.0f;
        uncertainty_color = Vec4f(1.0f, 0.8f * (1.0f - t), 0.0f, 1.0f);
    }

    for (auto iter = idx.begin(); iter != idx.end(); iter++) {
        const int& id = *iter;
        const auto& pt = cloud->points[id];
        auto pt_world = pose_l * cloud->points[id].getVector3fMap();
        xyz_data_[id] = Vec3f(pt_world.x(), pt_world.y(), pt_world.z());
        color_data_pcl_[id] = IntensityToRgbPCL(pt.intensity);
        color_data_gray_[id] = Vec4f(0.6, 0.65, 0.7, 1.0);
        color_data_uncertainty_[id] = uncertainty_color;
        color_data_height_[id] = IntensityToRgbPCL(pt.z * 10);
        color_data_intensity_[id] =
            Vec4f(pt.intensity / 255.0 * 3.0, pt.intensity / 255.0 * 3.0, pt.intensity / 255.0 * 3.0, 1.0);
    }
}

void UiCloud::Render() {
    // glPointSize(2.0);

    glBegin(GL_POINTS);

    for (int i = 0; i < xyz_data_.size(); ++i) {
        if (use_color_ == UseColor::PCL_COLOR) {
            glColor4f(color_data_pcl_[i][0], color_data_pcl_[i][1], color_data_pcl_[i][2], ui::opacity);
        } else if (use_color_ == UseColor::INTENSITY_COLOR) {
            glColor4f(color_data_intensity_[i][0], color_data_intensity_[i][1], color_data_intensity_[i][2],
                      ui::opacity);
        } else if (use_color_ == UseColor::HEIGHT_COLOR) {
            glColor4f(color_data_height_[i][0], color_data_height_[i][1], color_data_height_[i][2], ui::opacity);
        } else if (use_color_ == UseColor::GRAY_COLOR) {
            glColor4f(color_data_gray_[i][0], color_data_gray_[i][1], color_data_gray_[i][2], ui::opacity);
        } else if (use_color_ == UseColor::UNCERTAINTY_COLOR) {
            glColor4f(color_data_uncertainty_[i][0], color_data_uncertainty_[i][1], color_data_uncertainty_[i][2], ui::opacity);
        } else if (use_color_ == UseColor::CUSTOM_COLOR) {
            glColor4f(custom_color_[0], custom_color_[1], custom_color_[2], ui::opacity);
        }

        glVertex3f(xyz_data_[i][0], xyz_data_[i][1], xyz_data_[i][2]);
    }
    glEnd();
}

void UiCloud::BuildIntensityTable() {
    intensity_color_table_pcl_.reserve(255 * 3);
    // Modern viridis-inspired color palette for better visual appeal
    auto make_color = [](int r, int g, int b) -> Vec4f { return Vec4f(r / 255.0f, g / 255.0f, b / 255.0f, 0.7f); };

    // Viridis-inspired gradient: purple -> blue -> teal -> green -> yellow
    for (int i = 0; i < 256; i++) {
        // Purple to blue transition
        float t = i / 255.0f;
        int r = static_cast<int>(68 + t * (33 - 68));
        int g = static_cast<int>(1 + t * (145 - 1));
        int b = static_cast<int>(84 + t * (140 - 84));
        intensity_color_table_pcl_.emplace_back(make_color(r, g, b));
    }
    for (int i = 0; i < 256; i++) {
        // Blue to teal/green transition
        float t = i / 255.0f;
        int r = static_cast<int>(33 + t * (94 - 33));
        int g = static_cast<int>(145 + t * (201 - 145));
        int b = static_cast<int>(140 + t * (98 - 140));
        intensity_color_table_pcl_.emplace_back(make_color(r, g, b));
    }
    for (int i = 0; i < 256; i++) {
        // Green to yellow transition
        float t = i / 255.0f;
        int r = static_cast<int>(94 + t * (253 - 94));
        int g = static_cast<int>(201 + t * (231 - 201));
        int b = static_cast<int>(98 + t * (37 - 98));
        intensity_color_table_pcl_.emplace_back(make_color(r, g, b));
    }
}

void UiCloud::SetRenderColor(UiCloud::UseColor use_color) { use_color_ = use_color; }

}  // namespace lightning::ui
