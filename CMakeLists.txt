cmake_minimum_required(VERSION 2.6)
project(ASKER)
set (CMAKE_CXX_STANDARD 11)

set (V3D_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (V3D_INCLUDE_DIRS ${V3D_DIR}/.)

set (EXTRA_INC_DIRS /usr/local/include/eigen3/)
include (Config/v3d_macros.cmake)
include (Config/local_config.cmake)

include_directories(${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})

set (BASE_SRC
    Base/v3d_exception.h
    Base/v3d_serialization.h
    Base/v3d_vrmlio.h
    Base/v3d_image.cpp
)

set (MATH_SRC
    Math/v3d_linear.h
    Math/v3d_linearbase.h
    Math/v3d_mathutilities.h
    Math/v3d_nonlinlsq.h
    Math/v3d_nonlinlsq.cpp
    Math/v3d_optimization.h
    Math/v3d_optimization.cpp
    Math/v3d_optimization_lm.cpp
)

set (GEOMETRY_SRC
    Geometry/v3d_cameramatrix.h
    Geometry/v3d_distortion.h
    Geometry/v3d_metricbundle.h
    Geometry/v3d_metricbundle.cpp
    Geometry/v3d_stereobundle.h
    Geometry/v3d_stereobundle.cpp
)

set (ALL_SRC
    ${BASE_SRC} ${MATH_SRC} ${GEOMETRY_SRC} 
)

include_directories(${V3D_INCLUDE_DIRS} )

set (ASKER_MAIN_FILE bundle_large_filter.cpp)
set (ASKER_EXE asker)

set (MOO_MAIN_FILE bundle_large_moo.cpp)
set (MOO_EXE moo)

set (CMAKE_BUILD_TYPE Debug)
add_executable (${ASKER_EXE} Apps/${ASKER_MAIN_FILE} ${ALL_SRC})
target_link_libraries(${ASKER_EXE} colamd )
target_compile_options(${ASKER_EXE} PRIVATE -O3 )

add_executable (${MOO_EXE} Apps/${MOO_MAIN_FILE} ${ALL_SRC})
target_link_libraries(${MOO_EXE} colamd )
target_compile_options(${MOO_EXE} PRIVATE -O3 )



