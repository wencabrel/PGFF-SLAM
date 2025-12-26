#include "ui/imgui_panel.h"
#include <GL/glew.h>
#include <backends/imgui_impl_opengl3.h>
#include <glog/logging.h>

namespace lightning::ui {

ImGuiPanel::~ImGuiPanel() {
    if (initialized_) {
        Shutdown();
    }
}

bool ImGuiPanel::Init(int window_width, int window_height) {
    if (initialized_) {
        LOG(WARNING) << "[ImGui] Already initialized";
        return false;
    }

    window_width_ = window_width;
    window_height_ = window_height;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.IniFilename = nullptr;  // Disable imgui.ini file
    io.DisplaySize = ImVec2((float)window_width, (float)window_height);

    // Setup OpenGL3 renderer (standalone, no platform binding needed)
    // Pangolin already manages the OpenGL context
    ImGui_ImplOpenGL3_Init("#version 130");  // OpenGL 3.0+ GLSL 130

    // Apply custom theme
    ApplyCustomTheme();

    initialized_ = true;
    LOG(INFO) << "[ImGui] Initialized successfully (standalone OpenGL3 backend)";
    return true;
}

void ImGuiPanel::UpdateDisplaySize(int width, int height) {
    window_width_ = width;
    window_height_ = height;
}

void ImGuiPanel::NewFrame() {
    if (!initialized_) return;

    // Start the Dear ImGui frame (standalone - we handle input manually if needed)
    ImGui_ImplOpenGL3_NewFrame();
    
    // Manual frame setup since we're not using platform binding
    ImGuiIO& io = ImGui::GetIO();
    // Use the window dimensions that were set via UpdateDisplaySize
    io.DisplaySize = ImVec2((float)window_width_, (float)window_height_);
    io.DeltaTime = 1.0f / 60.0f;  // Assume 60 FPS
    
    ImGui::NewFrame();
}

void ImGuiPanel::Render() {
    if (!initialized_) return;

    // Main control panel window - same bounds as Pangolin right panel:
    // Pangolin right: SetBounds(0.0, 1.0, ...) = bottom 0% to top 100% of window
    // ImGui left: position at Y=0, height = full window height
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(280, (float)window_height_), ImGuiCond_Always);
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoMove | 
                                    ImGuiWindowFlags_NoResize |
                                    ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoTitleBar;

    if (ImGui::Begin("⚡ LIGHTNING SLAM", nullptr, window_flags)) {
        
        RenderViewControlsSection();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        RenderCameraPresetsSection();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        RenderPGFFModulesSection();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        RenderStatisticsSection();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        RenderMetricsSection();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        RenderPlaybackSection();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        RenderDisplaySection();
    }
    ImGui::End();

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiPanel::Shutdown() {
    if (!initialized_) return;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();

    initialized_ = false;
    LOG(INFO) << "[ImGui] Shutdown complete";
}

void ImGuiPanel::ApplyCustomTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Modern dark theme with vibrant accents
    colors[ImGuiCol_WindowBg]           = ImVec4(0.08f, 0.08f, 0.12f, 0.95f);  // Dark navy background
    colors[ImGuiCol_ChildBg]            = ImVec4(0.10f, 0.10f, 0.14f, 0.90f);
    colors[ImGuiCol_PopupBg]            = ImVec4(0.08f, 0.08f, 0.12f, 0.95f);
    colors[ImGuiCol_Border]             = ImVec4(0.20f, 0.25f, 0.35f, 0.60f);  // Steel blue border
    colors[ImGuiCol_BorderShadow]       = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]            = ImVec4(0.15f, 0.18f, 0.25f, 0.80f);  // Frame background
    colors[ImGuiCol_FrameBgHovered]     = ImVec4(0.20f, 0.25f, 0.35f, 0.90f);
    colors[ImGuiCol_FrameBgActive]      = ImVec4(0.25f, 0.30f, 0.40f, 1.00f);
    colors[ImGuiCol_TitleBg]            = ImVec4(0.05f, 0.10f, 0.20f, 1.00f);  // Deep blue title
    colors[ImGuiCol_TitleBgActive]      = ImVec4(0.10f, 0.20f, 0.35f, 1.00f);  // Brighter when active
    colors[ImGuiCol_TitleBgCollapsed]   = ImVec4(0.05f, 0.10f, 0.20f, 0.75f);
    colors[ImGuiCol_MenuBarBg]          = ImVec4(0.10f, 0.12f, 0.18f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]        = ImVec4(0.10f, 0.10f, 0.14f, 0.60f);
    colors[ImGuiCol_ScrollbarGrab]      = ImVec4(0.25f, 0.30f, 0.40f, 0.80f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.30f, 0.40f, 0.50f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.35f, 0.50f, 0.65f, 1.00f);
    colors[ImGuiCol_CheckMark]          = ImVec4(0.30f, 0.90f, 0.50f, 1.00f);  // Bright green checkmark
    colors[ImGuiCol_SliderGrab]         = ImVec4(0.35f, 0.70f, 0.90f, 1.00f);  // Cyan slider
    colors[ImGuiCol_SliderGrabActive]   = ImVec4(0.50f, 0.85f, 1.00f, 1.00f);  // Bright cyan active
    colors[ImGuiCol_Button]             = ImVec4(0.20f, 0.30f, 0.50f, 0.80f);  // Blue buttons
    colors[ImGuiCol_ButtonHovered]      = ImVec4(0.30f, 0.45f, 0.70f, 1.00f);  // Brighter on hover
    colors[ImGuiCol_ButtonActive]       = ImVec4(0.40f, 0.60f, 0.90f, 1.00f);  // Very bright when clicked
    colors[ImGuiCol_Header]             = ImVec4(0.20f, 0.30f, 0.45f, 0.80f);  // Headers (collapsing)
    colors[ImGuiCol_HeaderHovered]      = ImVec4(0.25f, 0.40f, 0.60f, 1.00f);
    colors[ImGuiCol_HeaderActive]       = ImVec4(0.30f, 0.50f, 0.75f, 1.00f);
    colors[ImGuiCol_Separator]          = ImVec4(0.20f, 0.25f, 0.35f, 0.70f);
    colors[ImGuiCol_SeparatorHovered]   = ImVec4(0.30f, 0.40f, 0.55f, 1.00f);
    colors[ImGuiCol_SeparatorActive]    = ImVec4(0.40f, 0.60f, 0.80f, 1.00f);
    colors[ImGuiCol_ResizeGrip]         = ImVec4(0.20f, 0.30f, 0.50f, 0.60f);
    colors[ImGuiCol_ResizeGripHovered]  = ImVec4(0.30f, 0.45f, 0.70f, 0.90f);
    colors[ImGuiCol_ResizeGripActive]   = ImVec4(0.40f, 0.60f, 0.90f, 1.00f);
    colors[ImGuiCol_Tab]                = ImVec4(0.15f, 0.20f, 0.35f, 0.90f);
    colors[ImGuiCol_TabHovered]         = ImVec4(0.25f, 0.40f, 0.65f, 1.00f);
    colors[ImGuiCol_TabActive]          = ImVec4(0.20f, 0.35f, 0.55f, 1.00f);
    colors[ImGuiCol_TabUnfocused]       = ImVec4(0.10f, 0.12f, 0.20f, 0.90f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.15f, 0.20f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotLines]          = ImVec4(0.40f, 0.80f, 1.00f, 1.00f);  // Cyan plot lines
    colors[ImGuiCol_PlotLinesHovered]   = ImVec4(0.60f, 0.90f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]      = ImVec4(0.50f, 0.70f, 0.30f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.70f, 0.90f, 0.50f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]     = ImVec4(0.25f, 0.45f, 0.70f, 0.60f);
    colors[ImGuiCol_DragDropTarget]     = ImVec4(0.50f, 0.80f, 1.00f, 0.90f);
    colors[ImGuiCol_NavHighlight]       = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]  = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]   = ImVec4(0.10f, 0.10f, 0.14f, 0.60f);

    // Style parameters
    style.WindowRounding    = 6.0f;
    style.ChildRounding     = 4.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 3.0f;
    style.TabRounding       = 4.0f;
    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
    style.TabBorderSize     = 0.0f;
    style.WindowPadding     = ImVec2(10.0f, 10.0f);
    style.FramePadding      = ImVec2(8.0f, 4.0f);
    style.ItemSpacing       = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);
    style.IndentSpacing     = 20.0f;
    style.ScrollbarSize     = 14.0f;
    style.GrabMinSize       = 10.0f;

    LOG(INFO) << "[ImGui] Custom theme applied";
}

void ImGuiPanel::RenderViewControlsSection() {
    if (ImGui::CollapsingHeader("🎥 View Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Follow Robot", &state_.follow_robot);
        ImGui::Checkbox("Frontend Traj (Red)", &state_.show_frontend_traj);
        ImGui::Checkbox("Backend Traj (Green)", &state_.show_backend_traj);
        ImGui::Checkbox("Loop Closures", &state_.show_loop_closures);
    }
}

void ImGuiPanel::RenderCameraPresetsSection() {
    if (ImGui::CollapsingHeader("📷 Camera Presets")) {
        if (ImGui::Button("Top-Down View", ImVec2(-1, 0))) {
            state_.btn_top_view = true;
        }
        if (ImGui::Button("Side View", ImVec2(-1, 0))) {
            state_.btn_side_view = true;
        }
        if (ImGui::Button("Follow View", ImVec2(-1, 0))) {
            state_.btn_follow_view = true;
        }
    }
}

void ImGuiPanel::RenderPGFFModulesSection() {
    if (ImGui::CollapsingHeader("⚡ PGFF Modules", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("PGFF Enabled", &state_.pgff_enabled);
        ImGui::Indent();
        ImGui::BeginDisabled(!state_.pgff_enabled);
        ImGui::Checkbox("Uncertainty Map", &state_.uncertainty_map);
        ImGui::Checkbox("Multi-Hyp LC", &state_.multi_hyp_lc);
        ImGui::Checkbox("Info Frontier", &state_.info_frontier);
        ImGui::Checkbox("Surprise Prior", &state_.surprise_prior);
        ImGui::EndDisabled();
        ImGui::Unindent();
    }
}

void ImGuiPanel::RenderStatisticsSection() {
    if (ImGui::CollapsingHeader("📊 Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Keyframes: %d", state_.keyframe_count);
        ImGui::Text("Loop Closures: %d", state_.loop_count);
        ImGui::Text("Distance: %.2f m", state_.distance_m);
        ImGui::Text("FPS: %.1f", state_.fps);
    }
}

void ImGuiPanel::RenderMetricsSection() {
    if (ImGui::CollapsingHeader("📈 PGFF Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Surprise: %.3f", state_.surprise);
        ImGui::Text("Uncertainty: %.3f m", state_.uncertainty_m);
        ImGui::Text("Info Accuracy: %.1f%%", state_.info_accuracy_pct);
        ImGui::Text("Active Hypotheses: %d", state_.active_hypotheses);
    }
}

void ImGuiPanel::RenderPlaybackSection() {
    if (ImGui::CollapsingHeader("▶️ Playback")) {
        ImGui::Checkbox("Step Mode", &state_.step_mode);
        ImGui::SliderFloat("Speed", &state_.play_speed, 0.1f, 10.0f, "%.1fx");
    }
}

void ImGuiPanel::RenderDisplaySection() {
    if (ImGui::CollapsingHeader("🎨 Display")) {
        ImGui::SliderFloat("Point Opacity", &state_.point_opacity, 0.1f, 1.0f, "%.2f");
        ImGui::SliderFloat("Point Size", &state_.point_size, 0.5f, 5.0f, "%.1f");
    }
}

} // namespace lightning::ui
