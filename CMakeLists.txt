cmake_minimum_required(VERSION 3.15)
project(AlphaZeroEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

# Add subdirectories
add_subdirectory(alphazero/core)
add_subdirectory(alphazero/bindings)