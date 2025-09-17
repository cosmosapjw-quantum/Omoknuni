# Unit tests for MCTS components

# Node pool tests
add_executable(test_node_pool
    test_node_pool.cpp
)

target_link_libraries(test_node_pool
    PRIVATE
    mcts_core
    gtest_main
    gtest
)

# Add compiler optimizations for realistic performance testing
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_node_pool PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_node_pool)

# PUCT selection tests
add_executable(test_selection
    test_selection.cpp
)

target_link_libraries(test_selection
    PRIVATE
    mcts_core
    gtest_main
    gtest
)

# Add compiler optimizations for realistic performance testing
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_selection PRIVATE -O2 -mavx2)
endif()

gtest_discover_tests(test_selection)

# Create a simple test runner for all unit tests
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel 4
    DEPENDS test_node_pool test_selection
    COMMENT "Running all unit tests"
)