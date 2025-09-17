# MCTS C++ extension module

# Create static library for MCTS components
add_library(mcts_core STATIC
    tree.cpp
    selection.cpp
)

# Set compile features and flags
target_compile_features(mcts_core PUBLIC cxx_std_17)
target_include_directories(mcts_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link with OpenMP
target_link_libraries(mcts_core PUBLIC OpenMP::OpenMP_CXX)

# Add compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mcts_core PRIVATE -mavx2 -mfma)
endif()

# Future pybind11 module (commented out for now)
# pybind11_add_module(mcts_python
#     python_bindings.cpp
# )
# target_link_libraries(mcts_python PRIVATE mcts_core)
# target_compile_definitions(mcts_python PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})