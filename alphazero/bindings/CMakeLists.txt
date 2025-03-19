pybind11_add_module(bindings
    game_bindings.cpp
)

target_include_directories(bindings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/game
)

target_link_libraries(bindings PRIVATE
    Threads::Threads
)

target_compile_features(bindings PRIVATE cxx_std_17)