cmake_minimum_required(VERSION 3.16)
project(autodiff_optimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(autodiff_optimization main.cpp)

if(MSVC)
  target_compile_options(autodiff_optimization PRIVATE /W4)
else()
  target_compile_options(autodiff_optimization PRIVATE -Wall -Wextra -Wpedantic)
endif()
