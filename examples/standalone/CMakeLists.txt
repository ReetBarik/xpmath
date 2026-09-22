# Standalone consumer of the INSTALLED xpmath package.
#
# This project is deliberately outside the main build. Configure it against an
# install prefix so find_package(xpmath) has to resolve the exported target:
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<xpmath-install-prefix>
#   cmake --build build
#   ./build/compensated_reduction
#
# No in-tree -Iinclude, no Kokkos, no device harness. The CUDA and HIP variants
# in this directory are separate source files with their own compile recipes.

cmake_minimum_required(VERSION 3.16)
project(xpmath_standalone_example LANGUAGES CXX)

find_package(xpmath REQUIRED)

add_executable(compensated_reduction compensated_reduction.cpp)
target_link_libraries(compensated_reduction PRIVATE xpmath::xpmath)
