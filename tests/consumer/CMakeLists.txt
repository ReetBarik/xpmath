# Consumer smoke test for the INSTALLED xpmath package.
#
# This is a STANDALONE project on purpose. It is never added via
# add_subdirectory() from the main build -- it is configured separately,
# pointed at an install prefix, so that find_package(xpmath) has to do real
# work. Adding it to the parent build would let it resolve xpmath::xpmath
# from the in-tree target and prove nothing about the install.
#
# Drive it with tests/consumer/run_consumer_test.sh.

cmake_minimum_required(VERSION 3.16)
project(xpmath_consumer_smoke LANGUAGES CXX)

# No version argument here: this must work against whatever is installed.
# The version FILE is exercised separately by run_consumer_test.sh, which
# asks for an impossible version and requires the lookup to fail.
find_package(xpmath REQUIRED)

add_executable(consumer_smoke consumer_smoke.cpp)
target_link_libraries(consumer_smoke PRIVATE xpmath::xpmath)

# Deliberately nothing else. No Kokkos, no libquadmath, no MPFR, no
# -fext-numeric-literals. If the installed headers need any of that to
# compile, this test is supposed to fail.
