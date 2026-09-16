// ===========================================================================
// validation/mi250/kokkos_space_probe.cpp — "did the execution space actually
//                                            resolve to HIP, or to Serial?"
// ===========================================================================
//
// WHY THIS FILE EXISTS AT ALL. The natural way to answer the question is to
// run one of the repo's own Kokkos-linked test binaries with
// --kokkos-print-configuration. On gfx90a that is not available: under the
// mi250 recipe NOT ONE of the 49 ctest targets links (see the S8c STATUS
// block), so there is no binary to ask. A question that cannot be asked is
// usually reported unanswered -- but "the build ran on the GPU" is load-
// bearing for every future MI250 claim, and leaving it open would mean the
// next session re-derives it. So: the smallest possible Kokkos program that
// still answers it honestly.
//
// It is deliberately NOT part of the test suite and not built by the repo's
// CMakeLists.txt. validation/mi250/run_mi250_build.sh compiles it in a
// throwaway CMake project against the SAME Kokkos install and the SAME
// toolchain the mi250 arch row selects, so what it reports is a property of
// that recipe, not of a hand-rolled compile line.
//
// WHAT IT ASSERTS, AND WHAT IT DOES NOT.
//   It asserts    : find_package(Kokkos) resolves under hipcc; the default
//                   execution space is HIP and not the Serial fallback; and a
//                   parallel_for in that space really executes, because the
//                   values come back off the device and are checked.
//   It does NOT   : say anything about extended-precision accuracy. Nothing
//                   here includes include/xp/. Do not cite it as device
//                   evidence for DD/FF/QF/TF.
//
// Two execution spaces are printed because a HIP-enabled Kokkos also has
// Serial enabled, and "SERIAL appears in the configuration" is not the same
// statement as "SERIAL is what the kernels ran in".
// ===========================================================================

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  int rc = 0;
  Kokkos::initialize(argc, argv);
  {
    const std::string dev  = Kokkos::DefaultExecutionSpace::name();
    const std::string host = Kokkos::DefaultHostExecutionSpace::name();

    // Machine-greppable, one fact per line. run_mi250_build.sh quotes these
    // verbatim into the log; the STATUS block quotes them from there.
    std::printf("resolved-default-execution-space: %s\n", dev.c_str());
    std::printf("resolved-default-host-execution-space: %s\n", host.c_str());

    // A name is a compile-time fact. Run something, so the line is backed by
    // an actual dispatch into that space.
    constexpr int N = 8;
    Kokkos::View<int*> v("probe", N);
    Kokkos::parallel_for(
        "space_probe", N, KOKKOS_LAMBDA(const int i) { v(i) = i * i; });
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);

    bool ok = true;
    for (int i = 0; i < N; ++i)
      if (h(i) != i * i) ok = false;
    std::printf("probe-kernel: %s (%d %d %d %d %d %d %d %d)\n",
                ok ? "CORRECT" : "WRONG", h(0), h(1), h(2), h(3), h(4), h(5),
                h(6), h(7));

    // The verdict line, so the script does not have to re-derive it from the
    // space name and get the spelling wrong on some future Kokkos.
    const bool is_hip = (dev == "HIP");
    std::printf("probe-verdict: %s\n",
                (is_hip && ok) ? "HIP-EXECUTED"
                               : (ok ? "NOT-HIP" : "KERNEL-WRONG"));
    if (!is_hip || !ok) rc = 1;

    std::printf("--- Kokkos::print_configuration ---\n");
    std::cout << std::flush;
    Kokkos::print_configuration(std::cout, /*detail=*/true);
    std::cout << std::flush;
  }
  Kokkos::finalize();
  return rc;
}
