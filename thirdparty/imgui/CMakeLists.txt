cmake_minimum_required(VERSION 3.16)

project(imgui)

set(CMAKE_CXX_STANDARD 17)

# ImGui core files
set(IMGUI_SOURCES
    imgui.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
)

# ImGui OpenGL3 backend ONLY (no GLFW - Pangolin manages window/context)
set(IMGUI_BACKEND_SOURCES
    backends/imgui_impl_opengl3.cpp
)

# Create ImGui library
add_library(imgui STATIC
    ${IMGUI_SOURCES}
    ${IMGUI_BACKEND_SOURCES}
)

# Find dependencies
find_package(OpenGL REQUIRED)

target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
    ${OPENGL_INCLUDE_DIR}
)

target_link_libraries(imgui PUBLIC
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
)

# Use GLEW as OpenGL loader (same as Pangolin)
target_compile_definitions(imgui PUBLIC
    IMGUI_IMPL_OPENGL_LOADER_GLEW
)

install(TARGETS imgui
    DESTINATION lib
)
