cmake_minimum_required(VERSION 3.20)
project(TTTN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tttn_test main.cpp)
target_include_directories(tttn_test PRIVATE src)

find_library(ACCELERATE_FRAMEWORK Accelerate)
if (ACCELERATE_FRAMEWORK)
    target_link_libraries(tttn_test PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(tttn_test PRIVATE ACCELERATE_NEW_LAPACK)
endif ()

# Sync docstrings from README.md into .hpp files before every build.
# Requires Python 3 on PATH. Pass -DSKIP_SYNC=ON to cmake to disable.
option(SKIP_SYNC "Skip README → docstring sync step" OFF)
if (NOT SKIP_SYNC)
    find_program(PYTHON3 python3 REQUIRED)
    add_custom_command(
            TARGET tttn_test PRE_BUILD
            COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/sync_docs.py ${CMAKE_SOURCE_DIR}/README.md
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Syncing docstrings from README.md..."
    )
endif ()
