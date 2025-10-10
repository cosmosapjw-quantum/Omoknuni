cmake_minimum_required(VERSION 3.14)
project(mcts_profiling)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# AVX2 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Source files for profiling library
set(PROFILING_SOURCES
    enhanced_profiler.cpp
)

# Create profiling library
add_library(mcts_profiling STATIC ${PROFILING_SOURCES})

# Link with parent MCTS library if it exists
if(TARGET mcts)
    target_link_libraries(mcts_profiling mcts)
endif()

# Profiling test executable
add_executable(profiling_test profiling_test.cpp)
target_link_libraries(profiling_test
    mcts_profiling
    pthread
)

# Google Test (optional)
find_package(GTest QUIET)
if(GTEST_FOUND)
    enable_testing()

    # Create test executable
    add_executable(profiling_gtest profiling_gtest.cpp)
    target_link_libraries(profiling_gtest
        mcts_profiling
        GTest::GTest
        GTest::Main
        pthread
    )

    # Add test
    add_test(NAME ProfilingTest COMMAND profiling_gtest)
else()
    message(STATUS "Google Test not found. Unit tests will not be built.")
endif()

# Installation
install(TARGETS mcts_profiling
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    enhanced_metrics.hpp
    enhanced_profiler.hpp
    thread_metrics.hpp
    statistical_analyzer.hpp
    DESTINATION include/mcts/profiling
)

# Export compile commands for tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)