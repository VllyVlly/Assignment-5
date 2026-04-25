cmake_minimum_required(VERSION 3.12)
project(code_opt_dev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# BONUS CMAKELISTS
# IF NEED TO USE, CHANGE NAME OF DEFAULT AND BONUS FILE

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
        -ffast-math -O3 -Wall -g -fno-omit-frame-pointer
)

set_source_files_properties(src/kernel/bitwise.cpp 
    PROPERTIES COMPILE_FLAGS "-ffast-math -Wall -g -fno-omit-frame-pointer -O3 -march=native -fprefetch-loop-arrays"
)

set_source_files_properties(src/kernel/matmul.cpp 
    PROPERTIES COMPILE_FLAGS "-ffast-math -Wall -g -fno-omit-frame-pointer -O3"
)

set_source_files_properties(src/kernel/trace_replay.cpp 
    PROPERTIES COMPILE_FLAGS "-ffast-math -Wall -g -fno-omit-frame-pointer -O3"
)

set_source_files_properties(src/kernel/blackscholes.cpp 
    PROPERTIES COMPILE_FLAGS "-ffast-math -Wall -g -fno-omit-frame-pointer -O3 -march=native"
)

set_source_files_properties(src/kernel/grff.cpp 
    PROPERTIES COMPILE_FLAGS "-ffast-math -Wall -g -fno-omit-frame-pointer -O3 -march=native -mtune=native"
)



target_include_directories(kernels PUBLIC include)

# ---- Executables ----

# add_executable(single_bench src/main/single_bench.cpp)
# target_link_libraries(single_bench PRIVATE kernels)

# add_executable(run_all src/main/run_all.cpp)
# target_link_libraries(run_all PRIVATE kernels)

# DEEPSEEK Suggestion for Windows
add_executable(single_bench src/main/single_bench.cpp)
target_link_libraries(single_bench kernels stdc++exp)

add_executable(run_all src/main/run_all.cpp)
target_link_libraries(run_all kernels stdc++exp)