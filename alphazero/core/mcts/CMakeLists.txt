set(MCTS_SOURCES
    mcts_node.cpp
    mcts.cpp
    transposition_table.cpp
)

add_library(mcts STATIC ${MCTS_SOURCES})
target_link_libraries(mcts PRIVATE utils)
target_include_directories(mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})