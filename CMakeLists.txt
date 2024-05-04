cmake_minimum_required(VERSION 3.28)
project(umbilic)

set(CMAKE_CXX_STANDARD 11)

add_executable(umbilic
        cpp_equal_corr/main.cpp
        cpp_equal_corr/main_single.cpp)
