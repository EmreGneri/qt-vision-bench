cmake_minimum_required(VERSION 3.21)
project(qt_vision_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The whole point of this repository is a timing comparison, and a Debug build
# would make those numbers meaningless. Single-config generators default to no
# build type at all, so pick Release when the caller did not choose.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

# For clangd, clang-tidy and friends.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

# Keep the warnings on. Applied only to our own targets; Qt and OpenCV headers
# arrive through -isystem, so their warnings do not add noise.
if (MSVC)
    set(PROJECT_WARNINGS /W4)
else()
    set(PROJECT_WARNINGS -Wall -Wextra -Wpedantic)
endif()

# CI builds with -DQVB_WERROR=ON, so a new warning fails the pipeline rather
# than quietly accumulating. Off locally, where a warning should not block work.
option(QVB_WERROR "Treat compiler warnings as errors" OFF)
if (QVB_WERROR)
    if (MSVC)
        list(APPEND PROJECT_WARNINGS /WX)
    else()
        list(APPEND PROJECT_WARNINGS -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# vision_core: the image processing pipeline. Deliberately free of Qt so that
# the headless benchmark and the unit tests can run without a display.
# ---------------------------------------------------------------------------
add_library(vision_core STATIC
    src/pipeline.cpp
    src/pipeline.h
)
target_include_directories(vision_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})
target_compile_options(vision_core PRIVATE ${PROJECT_WARNINGS})

# ---------------------------------------------------------------------------
# The application. No WIN32 flag on purpose: --bench prints JSON to stdout, and
# capturing that without a console window would be awkward.
# ---------------------------------------------------------------------------
add_executable(qt_vision_bench
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.h
    src/worker.cpp
    src/worker.h
)
target_link_libraries(qt_vision_bench PRIVATE vision_core Qt6::Widgets)
target_compile_options(qt_vision_bench PRIVATE ${PROJECT_WARNINGS})

# The benchmark output carries the build type, so a Release-only result cannot
# silently come from a Debug binary. bench/sweep.py checks the field.
target_compile_definitions(qt_vision_bench PRIVATE QVB_BUILD_TYPE="$<CONFIG>")

# Only this target contains Q_OBJECT classes, so only this one needs the Qt code
# generators. Running them over the Qt-free library and the tests would just add
# an empty translation unit to every build.
set_target_properties(qt_vision_bench PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE vision_core)
target_compile_options(test_pipeline PRIVATE ${PROJECT_WARNINGS})

add_test(NAME pipeline COMMAND test_pipeline)

# ---------------------------------------------------------------------------
# Install: enough to run the executable from a staging directory.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS qt_vision_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
