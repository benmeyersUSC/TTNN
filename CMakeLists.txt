cmake_minimum_required(VERSION 3.20)
project(TTTN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tttn_test main.cpp)
target_include_directories(tttn_test PRIVATE src)
