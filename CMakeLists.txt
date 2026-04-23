cmake_minimum_required(VERSION 3.12)
project(code_opt_dev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/kernel/*.cpp"
)

add_library(kernels STATIC ${KERNEL_SOURCES})

target_include_directories(kernels
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

target_compile_options(kernels
    PRIVATE
        -ffast-math -O2 -Wall -g -fno-omit-frame-pointer
)
target_include_directories(kernels PUBLIC include)

# ---- Executables ----

add_executable(single_bench src/main/single_bench.cpp)
target_link_libraries(single_bench PRIVATE kernels)

add_executable(run_all src/main/run_all.cpp)
target_link_libraries(run_all PRIVATE kernels)