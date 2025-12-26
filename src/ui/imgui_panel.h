#pragma once

#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <memory>
#include <functional>

namespace lightning::ui {

/**
 * @brief ImGui panel manager for SLAM UI
 * 
 * This class encapsulates ImGui initialization, rendering, and lifecycle management.
 * It integrates with Pangolin's OpenGL context to render custom UI panels with
 * full styling control, replacing Pangolin's hardcoded menu widgets.
 * 
 * NOTE: Uses custom OpenGL3 backend without GLFW since Pangolin manages the window
 */
class ImGuiPanel {
public:
    ImGuiPanel() = default;
    ~ImGuiPanel();

    /**
     * @brief Initialize ImGui with OpenGL3 backend (no GLFW, uses Pangolin's context)
     * @param window_width Window width for sizing
     * @param window_height Window height for sizing
     * @return true if initialization successful
     */
    bool Init(int window_width, int window_height);

    /**
     * @brief Update display size (call when window is resized)
     */
    void UpdateDisplaySize(int width, int height);

    /**
     * @brief Begin new ImGui frame - call before rendering widgets
     */
    void NewFrame();

    /**
     * @brief Render ImGui frame - call after all widgets defined
     */
    void Render();

    /**
     * @brief Shutdown ImGui and cleanup resources
     */
    void Shutdown();

    /**
     * @brief Apply custom dark theme with vibrant accents
     */
    void ApplyCustomTheme();

    /**
     * @brief Check if ImGui is initialized
     */
    bool IsInitialized() const { return initialized_; }

    // UI State Variables (replaces pangolin::Var)
    struct UIState {
        // View Controls
        bool follow_robot = false;
        bool show_frontend_traj = true;
        bool show_backend_traj = false;
        bool show_loop_closures = true;

        // Camera Presets (buttons trigger actions)
        bool btn_top_view = false;
        bool btn_side_view = false;
        bool btn_follow_view = false;

        // PGFF Module Toggles
        bool pgff_enabled = true;
        bool uncertainty_map = true;
        bool multi_hyp_lc = true;
        bool info_frontier = true;
        bool surprise_prior = true;

        // Statistics (read-only display)
        int keyframe_count = 0;
        int loop_count = 0;
        double distance_m = 0.0;
        double fps = 0.0;

        // PGFF Metrics (read-only display)
        double surprise = 0.0;
        double uncertainty_m = 0.0;
        double info_accuracy_pct = 0.0;
        int active_hypotheses = 0;

        // Playback Controls
        bool step_mode = false;
        float play_speed = 10.0f;

        // Display Settings
        float point_opacity = 0.7f;
        float point_size = 1.0f;
    };

    UIState& GetState() { return state_; }
    const UIState& GetState() const { return state_; }

private:
    bool initialized_ = false;
    int window_width_ = 1920;
    int window_height_ = 1080;
    UIState state_;

    // Helper functions to render different panel sections
    void RenderViewControlsSection();
    void RenderCameraPresetsSection();
    void RenderPGFFModulesSection();
    void RenderStatisticsSection();
    void RenderMetricsSection();
    void RenderPlaybackSection();
    void RenderDisplaySection();
};

} // namespace lightning::ui
