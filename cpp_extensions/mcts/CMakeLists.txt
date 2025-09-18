# MCTS C++ extension module
cmake_minimum_required(VERSION 3.15)
project(mcts_core)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create static library for MCTS components
add_library(mcts_core STATIC
    tree.cpp
    tree.hpp
    selection.cpp
    selection.hpp
    virtual_loss.cpp
    virtual_loss.hpp
    backup.cpp
    backup.hpp
)

# Set compile features and flags
target_compile_features(mcts_core PUBLIC cxx_std_17)
target_include_directories(mcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with OpenMP
target_link_libraries(mcts_core PUBLIC OpenMP::OpenMP_CXX)

# Add compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mcts_core PRIVATE
        -Wall -Wextra
        -march=native  # Use native CPU architecture features
        -mavx2         # Enable AVX2 for vectorized PUCT selection
        -mfma          # Enable FMA instructions
        -fopenmp
    )
endif()

# Add compile definitions
target_compile_definitions(mcts_core PRIVATE
    _REENTRANT
    ALIGNMENT=64
)

# Enable position independent code for Python extension
set_target_properties(mcts_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Minimal compilation test
add_executable(mcts_compile_test
    tree.cpp
    selection.cpp
    virtual_loss.cpp
    backup.cpp
)

# Create minimal main function for compilation test
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test_main.cpp"
"#include \"tree.hpp\"
#include \"selection.hpp\"
#include \"virtual_loss.hpp\"
#include \"backup.hpp\"
#include <iostream>

int main() {
    std::cout << \"MCTS components compiled successfully!\" << std::endl;
    
    // Basic instantiation test
    mcts::MCTSTree tree(1000);
    mcts::VirtualLossConfig vl_config;
    mcts::BackupConfig backup_config;
    
    std::cout << \"Tree nodes: \" << tree.get_node_count() << std::endl;
    std::cout << \"All components instantiated successfully!\" << std::endl;
    return 0;
}
")

target_sources(mcts_compile_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/test_main.cpp")
target_include_directories(mcts_compile_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcts_compile_test OpenMP::OpenMP_CXX pthread)

# Add AVX2 support to test executable
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mcts_compile_test PRIVATE
        -Wall -Wextra
        -march=native  # Use native CPU architecture features
        -mavx2         # Enable AVX2 for vectorized PUCT selection
        -mfma          # Enable FMA instructions
        -fopenmp
    )
endif()

# Register compilation test
add_test(NAME mcts_compilation COMMAND mcts_compile_test)

message(STATUS "MCTS core library configured with all components")
