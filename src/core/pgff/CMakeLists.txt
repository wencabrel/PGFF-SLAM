# PGFF - Predictive Geometric Flow Fields
# Groundbreaking SLAM enhancement module

set(PGFF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/flow_field.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/surprise_detector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/predictive_lio.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/pgff_loop_closing.cc
)

set(PGFF_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/pgff.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/flow_field.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/jacobian_cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/surprise_detector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/predictive_lio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/enhanced_obs_model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/surprise_loop_detector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pgff/pgff_loop_closing.h
)

# Export for parent CMakeLists
set(PGFF_SOURCES ${PGFF_SOURCES} PARENT_SCOPE)
set(PGFF_HEADERS ${PGFF_HEADERS} PARENT_SCOPE)
