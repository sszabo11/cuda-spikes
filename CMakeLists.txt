cmake_minimum_required(VERSION 3.18)

project(Spikes LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(spikes main.c spike.cu data.h utils.c config.h process.c data.c render.h)

enable_testing()
add_executable(test_main test.c spike.cu data.h utils.c config.h process.c data.c render.c)

set_target_properties(spikes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"          # Or "80;86;89;90" for specific GPUs
)

include(FetchContent)
FetchContent_Declare(
  raylib
  GIT_REPOSITORY https://github.com/raysan5/raylib.git
  GIT_TAG master # Or a specific version like "5.0"
)
FetchContent_MakeAvailable(raylib)

target_link_libraries(spikes PRIVATE CUDA::cudart)
target_link_libraries(test_main PRIVATE CUDA::cudart raylib)
#set_target_properties(spikes PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 

target_include_directories(spikes PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA toolkit include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
