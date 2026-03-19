cmake_minimum_required(VERSION 3.20)
project(TTNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ttnn_test main.cpp)
target_include_directories(ttnn_test PRIVATE src)
