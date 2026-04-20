cmake_minimum_required(VERSION 3.18)

set(CUDAToolkit_ROOT /opt/cuda)
#set(CUDACXX /opt/cuda/bin/nvcc)
project(Spikes LANGUAGES C CXX CUDA)

set(raylib_DIR "${CMAKE_BINARY_DIR}/_deps/raylib-build" CACHE PATH "Path to raylib config" FORCE)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_BUILD_TYPE release)

file(GLOB_RECURSE C_SOURCES    CONFIGURE_DEPENDS "libs/**/*.c" "src/*.c")
file(GLOB_RECURSE CU_SOURCES   CONFIGURE_DEPENDS "libs/**/*.cu")

add_library(snn_c    STATIC ${C_SOURCES})
add_library(snn_cuda STATIC ${CU_SOURCES})

#add_executable(spikes main.c spike.cu data.h utils.c config.h process.c data.c render.h)

enable_testing()
#add_executable(test_main test.c spike.cu data.h utils.c config.h process.c data.c render.c)

target_include_directories(snn_c    PUBLIC include/)
target_include_directories(snn_cuda PUBLIC include/)
target_include_directories(snn_c PRIVATE 
    ${CMAKE_BINARY_DIR}/_deps/raylib-src/src
)


set_target_properties(snn_c PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"          # Or "80;86;89;90" for specific GPUs
)

set(SUPPORT_FILEFORMAT_JPG, 1)

include(FetchContent)
FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG master # Or a specific version like "5.0"
)
FetchContent_MakeAvailable(raylib)

#set_target_properties(spikes PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 

target_include_directories(snn_c PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(main src/main.c)
target_link_libraries(main PRIVATE snn_c snn_cuda)

add_executable(foo src/foo.c)
target_link_libraries(foo PRIVATE snn_c snn_cuda)

add_executable(test_main src/test.c)
target_link_libraries(test_main PRIVATE snn_c snn_cuda)

target_link_libraries(main PRIVATE CUDA::cudart)
target_link_libraries(test_main PRIVATE CUDA::cudart raylib)
target_link_libraries(foo PRIVATE CUDA::cudart raylib)


message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA toolkit include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
