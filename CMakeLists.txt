cmake_minimum_required(VERSION 3.18)

project(Spikes LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(spikes main.c spike.cu process.c data.h)

set_target_properties(spikes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"          # Or "80;86;89;90" for specific GPUs
)

target_link_libraries(spikes PRIVATE CUDA::cudart)
#set_target_properties(spikes PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 

target_include_directories(spikes PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA toolkit include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
