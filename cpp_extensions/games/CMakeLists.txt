# Games C++ extension module

# Collect all game source files
set(GAMES_SOURCES
    # Chess implementation
    chess/chess_state.cpp
    chess/chess_rules.cpp
    chess/chess960.cpp

    # Gomoku implementation
    gomoku/gomoku_state.cpp
    gomoku/gomoku_rules.cpp

    # Go implementation
    go/go_state.cpp
    go/go_rules.cpp

    # Game adapter interface
    interface.cpp

    # Python bindings
    python_bindings.cpp
)

# Create pybind11 module
pybind11_add_module(games_core ${GAMES_SOURCES})

# Link libraries
target_link_libraries(games_core PRIVATE
    OpenMP::OpenMP_CXX
    utils_core
)

# Include directories
target_include_directories(games_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp_extensions/utils
)

# Compiler definitions
target_compile_definitions(games_core PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
    PYBIND11_DETAILED_ERROR_MESSAGES
)