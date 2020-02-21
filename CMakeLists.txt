cmake_minimum_required(VERSION 3.13)
project(dynamo)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-O3 -Werror -Wall -Wextra")
include_directories (include)
include_directories (include/eigen)

add_executable (test_train src/test_train.cc)
add_executable (inference src/inference.cc)
add_executable (test src/test.cc)
add_executable (test_load src/test_load.cc)
add_executable (finite_diff_test src/finite_diff_test.cc)
