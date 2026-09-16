// ===========================================================================
// derecursion_runtime.hip — RUNTIME verification of the gfx90a de-recursion fix
// ===========================================================================
//
// docs/ROCM_RECURSIVE_DEVICE_STACK.md's verification table is entirely static
// or host-side: emitted-asm self-call counts, uses_dynamic_stack, the lint exit
// code, and host bit-identity over 90,453 points. What has never been done is
// running the reflected arms ON THE GPU after the fix. The doc says so for the
// complex kernels: "post-fix behaviour is asserted only structurally."
//
// This closes that loop for the nine functions that carried a device self-call
// at 6ddcb2d:  asinh (DD/FF/QF/TF), tanh (DD/FF/QF), tgamma (DD/FF).
//
// PRE-FIX SYMPTOMS, which this must show to be gone:
//   * FF and TF  -> hipErrorIllegalAddress
//   * DD and QF  -> silently WRONG values on the reflected (negative) arm
// One is loud, one is silent, so the test checks both: HIP error status after
// every launch, AND a bit-exact device-vs-host comparison of every limb.
//
// RUN AT THE DEFAULT STACK. The defect only appears at the HIP runtime's
// default per-thread allowance (measured 1024 B); raising it hides the bug.
// So pass 1 never touches hipLimitStackSize, and only if something fails does
// pass 2 retry at 64 KB -- because "fails at default, passes raised" is the
// signature of the stack defect specifically, as opposed to any other
// divergence. That distinction is worth more than a bare pass/fail.
// ===========================================================================

#include <hip/hip_runtime.h>
#include <cstdio>

#include <xp/dd_math.hpp>
#include <xp/ff_math.hpp>
#include <xp/qf_math.hpp>
#include <xp/tf_math.hpp>
#include <xp/dd_complex.hpp>
#include <xp/ff_complex.hpp>
#include <xp/qf_complex.hpp>
#include <xp/tf_complex.hpp>

// Limb extractors. float -> double is exact, so transporting every limb as a
// double keeps the comparison bit-exact rather than merely close.
XPMATH_INLINE_FUNCTION void words(const xp::DoubleDouble& v, double* o) { o[0]=v.hi; o[1]=v.lo; }
XPMATH_INLINE_FUNCTION void words(const xp::FloatFloat&   v, double* o) { o[0]=v.hi; o[1]=v.lo; }
XPMATH_INLINE_FUNCTION void words(const xp::QuadFloat&    v, double* o) { o[0]=v.f0; o[1]=v.f1; o[2]=v.f2; o[3]=v.f3; }
XPMATH_INLINE_FUNCTION void words(const xp::TripleFloat&  v, double* o) { o[0]=v.f0; o[1]=v.f1; o[2]=v.f2; }

// Complex results are laid out [re limbs..., im limbs...].
XPMATH_INLINE_FUNCTION void words(const xp::DoubleDoubleComplex& v, double* o) { words(v.re,o);   words(v.im,o+2); }
XPMATH_INLINE_FUNCTION void words(const xp::FloatFloatComplex&   v, double* o) { words(v.re,o);   words(v.im,o+2); }
XPMATH_INLINE_FUNCTION void words(const xp::QuadFloatComplex&    v, double* o) { words(v.re,o);   words(v.im,o+4); }
XPMATH_INLINE_FUNCTION void words(const xp::TripleFloatComplex&  v, double* o) { words(v.re,o);   words(v.im,o+3); }

#define GEN(tag, TYPE, SCALAR, NW, FN)                                         \
  __global__ void kern_##tag(const double* in, double* out, int n) {           \
      int i = blockIdx.x * blockDim.x + threadIdx.x;                           \
      if (i < n) words(xp::FN(xp::TYPE((SCALAR)in[i])), out + (size_t)i * (NW)); \
  }                                                                            \
  static void launch_##tag(const double* d_in, double* d_out, int n) {         \
      hipLaunchKernelGGL(kern_##tag, dim3((n + 63) / 64), dim3(64), 0, 0,      \
                         d_in, d_out, n);                                      \
  }                                                                            \
  static void host_##tag(const double* in, double* out, int n) {               \
      for (int i = 0; i < n; ++i)                                              \
          words(xp::FN(xp::TYPE((SCALAR)in[i])), out + (size_t)i * (NW));      \
  }

// Complex variant. docs/ROCM_RECURSIVE_DEVICE_STACK.md: the ComplexFunctor
// kernels also carried uses_dynamic_stack=1 pre-fix, "by reaching the real
// asinh and tanh through their complex counterparts" -- and no complex arm was
// ever measured on device. These cases are what closes that.
#define GENC(tag, CTYPE, RTYPE, SCALAR, NW2, FN)                               \
  __global__ void kern_##tag(const double* in, double* out, int n) {           \
      int i = blockIdx.x * blockDim.x + threadIdx.x;                           \
      if (i < n) {                                                             \
          xp::CTYPE z{xp::RTYPE((SCALAR)in[2*i]), xp::RTYPE((SCALAR)in[2*i+1])};\
          words(xp::FN(z), out + (size_t)i * (NW2));                           \
      }                                                                        \
  }                                                                            \
  static void launch_##tag(const double* d_in, double* d_out, int n) {         \
      hipLaunchKernelGGL(kern_##tag, dim3((n + 63) / 64), dim3(64), 0, 0,      \
                         d_in, d_out, n);                                      \
  }                                                                            \
  static void host_##tag(const double* in, double* out, int n) {               \
      for (int i = 0; i < n; ++i) {                                            \
          xp::CTYPE z{xp::RTYPE((SCALAR)in[2*i]), xp::RTYPE((SCALAR)in[2*i+1])};\
          words(xp::FN(z), out + (size_t)i * (NW2));                           \
      }                                                                        \
  }

// The nine self-calls removed at 33e4767: asinh x4, tanh x3, tgamma x2.
GEN(dd_asinh,  DoubleDouble, double, 2, asinh)
GEN(ff_asinh,  FloatFloat,   float,  2, asinh)
GEN(qf_asinh,  QuadFloat,    float,  4, asinh)
GEN(tf_asinh,  TripleFloat,  float,  3, asinh)
GEN(dd_tanh,   DoubleDouble, double, 2, tanh)
GEN(ff_tanh,   FloatFloat,   float,  2, tanh)
GEN(qf_tanh,   QuadFloat,    float,  4, tanh)
GEN(dd_tgamma, DoubleDouble, double, 2, tgamma)
GEN(ff_tgamma, FloatFloat,   float,  2, tgamma)

// The complex arms that reach them.
GENC(cdd_asinh, DoubleDoubleComplex, DoubleDouble, double, 4, asinh)
GENC(cff_asinh, FloatFloatComplex,   FloatFloat,   float,  4, asinh)
GENC(cqf_asinh, QuadFloatComplex,    QuadFloat,    float,  8, asinh)
GENC(ctf_asinh, TripleFloatComplex,  TripleFloat,  float,  6, asinh)
GENC(cdd_tanh,  DoubleDoubleComplex, DoubleDouble, double, 4, tanh)
GENC(cff_tanh,  FloatFloatComplex,   FloatFloat,   float,  4, tanh)
GENC(cqf_tanh,  QuadFloatComplex,    QuadFloat,    float,  8, tanh)
GENC(ctf_tanh,  TripleFloatComplex,  TripleFloat,  float,  6, tanh)

static double ulp_delta_c(const double* got, const double* ref, int cw, int p) {
    double dre = 0.0, dim = 0.0;
    for (int w = 0; w < cw; ++w)      dre += got[w] - ref[w];
    for (int w = cw; w < 2 * cw; ++w) dim += got[w] - ref[w];
    if (dre < 0.0) dre = -dre;
    if (dim < 0.0) dim = -dim;
    double r0 = ref[0]      < 0.0 ? -ref[0]      : ref[0];
    double i0 = ref[cw]     < 0.0 ? -ref[cw]     : ref[cw];
    double scale = r0 > i0 ? r0 : i0;
    if (scale == 0.0) return (dre + dim) == 0.0 ? 0.0 : 1.0e300;
    double ulp = scale;
    for (int i = 0; i < p; ++i) ulp *= 0.5;
    return (dre + dim) / ulp;
}

struct Case {
    const char* name;
    int         nw;
    int         p;            // format precision, for the ulp verdict
    int         cw;           // 0 = real; else limbs per complex component
    void      (*launch)(const double*, double*, int);
    void      (*host)(const double*, double*, int);
    int         set;          // 0 = odd-function inputs, 1 = tgamma inputs
};

static const Case kCases[] = {
    {"asinh  DD", 2, 106, 0, launch_dd_asinh,  host_dd_asinh,  0},
    {"asinh  FF", 2, 48, 0, launch_ff_asinh,  host_ff_asinh,  0},
    {"asinh  QF", 4, 96, 0, launch_qf_asinh,  host_qf_asinh,  0},
    {"asinh  TF", 3, 72, 0, launch_tf_asinh,  host_tf_asinh,  0},
    {"tanh   DD", 2, 106, 0, launch_dd_tanh,   host_dd_tanh,   0},
    {"tanh   FF", 2, 48, 0, launch_ff_tanh,   host_ff_tanh,   0},
    {"tanh   QF", 4, 96, 0, launch_qf_tanh,   host_qf_tanh,   0},
    {"tgamma DD", 2, 106, 0, launch_dd_tgamma, host_dd_tgamma, 1},
    {"tgamma FF", 2, 48, 0, launch_ff_tgamma, host_ff_tgamma, 1},
    {"c asinh DD", 4, 106, 2, launch_cdd_asinh, host_cdd_asinh, 2},
    {"c asinh FF", 4,  48, 2, launch_cff_asinh, host_cff_asinh, 2},
    {"c asinh QF", 8,  96, 4, launch_cqf_asinh, host_cqf_asinh, 2},
    {"c asinh TF", 6,  72, 3, launch_ctf_asinh, host_ctf_asinh, 2},
    {"c tanh  DD", 4, 106, 2, launch_cdd_tanh,  host_cdd_tanh,  2},
    {"c tanh  FF", 4,  48, 2, launch_cff_tanh,  host_cff_tanh,  2},
    {"c tanh  QF", 8,  96, 4, launch_cqf_tanh,  host_cqf_tanh,  2},
    {"c tanh  TF", 6,  72, 3, launch_ctf_tanh,  host_ctf_tanh,  2},
};
static const int kNumCases = (int)(sizeof(kCases) / sizeof(kCases[0]));

// Negatives first and in the majority: the reflected arm is the whole point.
// A grid that is symmetric about zero at equal magnitude is what makes a
// sign-fold bug visible at all.
static const double kOdd[] = {
    -1000.0, -10.0, -2.0, -1.0, -0.5, -0.25, -1e-3, -0.0,
     0.0, 1e-3, 0.25, 0.5, 1.0, 2.0, 10.0, 1000.0
};
// tgamma reflects below 0.5; poles sit on the non-positive integers, so the
// negative samples are deliberately half-integers and irrationals, never -1,-2.
static const double kGam[] = {
    -4.7, -3.5, -2.5, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5, 3.7, 6.0
};
static const double kCplx[] = {
    -3.0,-2.0,  -2.0, 0.5,  -1.0, 0.25, -0.5,-0.75, -0.25, 0.1, -1.5,-1.25,
     3.0, 2.0,   2.0, 0.5,   1.0,-0.25,  0.5, 0.75,  0.25,-0.1,  1.5, 1.25
};
static const int kNCplx = (int)(sizeof(kCplx) / sizeof(kCplx[0])) / 2;
static const int kNOdd = (int)(sizeof(kOdd) / sizeof(kOdd[0]));
static const int kNGam = (int)(sizeof(kGam) / sizeof(kGam[0]));

static bool bits_equal(double a, double b) {
    unsigned long long ua, ub;
    __builtin_memcpy(&ua, &a, 8);
    __builtin_memcpy(&ub, &b, 8);
    return ua == ub;
}

// THE VERDICT CRITERION, AND WHY IT IS NOT BIT-EXACTNESS.
//
// The first version of this test failed a case on any bit difference. Job
// 1000932 then "failed" 4 of 9 -- and scoring those 4 against a binary128
// oracle showed asinh QF/TF differing by 0.04 ulp, and tgamma DD/FF where the
// DEVICE WAS MORE ACCURATE THAN THE HOST (325 vs 327 ulp, 11.7 vs 15.5 ulp).
// Nothing was wrong with the library; the assertion was wrong.
//
// Host and device do not share a scalar math library -- the host pass uses
// glibc, the device pass ROCm's OCML, and neither is required to be correctly
// rounded for log/exp/sqrt. That seed difference lands in the LAST limb of the
// expansion, which is exactly where all four differences appeared. Bit
// equality across two different math libraries was never achievable, and
// demanding it makes host behaviour the specification -- the same mistake
// docs/CORRECTNESS.md retired the per-op tolerance tables for.
//
// So the criterion is the repo's own: relative difference measured in ulps of
// the format. The defect this test exists to catch produced grossly wrong
// values on the reflected arm, orders of magnitude above this threshold.
//
// The difference is computed limb-by-limb and summed in double. Every limb
// above the last is bit-identical in practice, so the sum is exactly the
// last-limb difference and carries no rounding of its own -- which is why this
// does not need a wider type it could not have on device anyway.
static const double kUlpTol = 4.0;

static double ulp_delta(const double* got, const double* ref, int nw, int p) {
    double diff = 0.0, scale = (ref[0] < 0.0) ? -ref[0] : ref[0];
    for (int w = 0; w < nw; ++w) diff += got[w] - ref[w];
    if (diff < 0.0) diff = -diff;
    if (scale == 0.0) return diff == 0.0 ? 0.0 : 1.0e300;
    double ulp = scale;
    for (int i = 0; i < p; ++i) ulp *= 0.5;   // scale * 2^-p, no libm needed
    return diff / ulp;
}

// Returns 0 = pass, 1 = value mismatch, 2 = HIP error.
static int run_case(const Case& c, bool verbose) {
    const double* in = (c.set == 0) ? kOdd : (c.set == 1) ? kGam : kCplx;
    const int     n  = (c.set == 0) ? kNOdd : (c.set == 1) ? kNGam : kNCplx;
    // Complex points occupy two doubles each.
    const size_t  nb_in  = (size_t)n * (c.cw ? 2 : 1) * sizeof(double);
    const size_t  nb_out = (size_t)n * c.nw * sizeof(double);

    double *d_in = nullptr, *d_out = nullptr;
    if (hipMalloc(&d_in, nb_in) != hipSuccess ||
        hipMalloc(&d_out, nb_out) != hipSuccess) {
        std::printf("      hipMalloc FAILED\n");
        return 2;
    }
    hipMemcpy(d_in, in, nb_in, hipMemcpyHostToDevice);
    hipMemset(d_out, 0, nb_out);

    c.launch(d_in, d_out, n);

    // Two checks, not one: hipGetLastError catches a bad launch, and the
    // synchronize is what surfaces hipErrorIllegalAddress from inside the
    // kernel -- the loud arm of this defect.
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        std::printf("      HIP ERROR: %s (%d)\n", hipGetErrorString(e), (int)e);
        hipFree(d_in); hipFree(d_out);
        return 2;
    }

    double* got = (double*)std::malloc(nb_out);
    double* ref = (double*)std::malloc(nb_out);
    hipMemcpy(got, d_out, nb_out, hipMemcpyDeviceToHost);
    c.host(in, ref, n);

    int    nbit = 0, nover = 0;
    double worst = 0.0, worst_x = 0.0;
    for (int i = 0; i < n; ++i) {
        const size_t base = (size_t)i * c.nw;
        for (int w = 0; w < c.nw; ++w)
            if (!bits_equal(got[base + w], ref[base + w])) { ++nbit; break; }

        double u = c.cw ? ulp_delta_c(got + base, ref + base, c.cw, c.p)
                        : ulp_delta(got + base, ref + base, c.nw, c.p);
        if (u > worst) { worst = u; worst_x = c.cw ? in[2*i] : in[i]; }
        if (u > kUlpTol) {
            ++nover;
            if (nover <= 4 || verbose) {
                if (c.cw) std::printf("      OVER TOL z=(% g,% g)  %.3f ulp  (tol %.1f)\n",
                                      in[2*i], in[2*i+1], u, kUlpTol);
                else      std::printf("      OVER TOL x=% -10.6g  %.3f ulp  (tol %.1f)\n",
                                      in[i], u, kUlpTol);
            }
        }
    }
    std::free(got); std::free(ref);
    hipFree(d_in); hipFree(d_out);

    if (nover) {
        std::printf("      %d/%d points above %.1f ulp  <-- the SILENT arm of the defect\n",
                    nover, n, kUlpTol);
        return 1;
    }
    // Bit differences below tolerance are REPORTED, not failed: they are the
    // host/device scalar-math-library difference, and seeing the margin every
    // run is worth more than a bare "ok".
    if (nbit) std::printf("ok  (%d/%d pts differ in the last limb, worst %.3f ulp at %g)\n",
                          nbit, n, worst, worst_x);
    else      std::printf("ok  (bit-identical, %d pts)\n", n);
    return 0;
}

int main() {
    int dev = 0;
    hipDeviceProp_t prop;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        std::printf("FATAL: no HIP device\n");
        return 2;
    }
    size_t stack = 0;
    hipDeviceGetLimit(&stack, hipLimitStackSize);

    std::printf("=== gfx90a de-recursion RUNTIME verification ===\n");
    std::printf("device        : %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("stack limit   : %zu B  (DEFAULT -- not raised; raising it hides the defect)\n", stack);
    std::printf("cases         : %d (9 real arms that carried a device self-call at\n"
                "                6ddcb2d, plus the 8 complex arms that reach them)\n", kNumCases);
    std::printf("verdict       : device-vs-host within %.1f ulp of the format -- NOT\n"
                "                bit-exactness; the two use different scalar math\n"
                "                libraries. See the note above kUlpTol.\n\n", kUlpTol);

    int fail_val = 0, fail_hip = 0;
    int failed[32]; int nfailed = 0;

    for (int i = 0; i < kNumCases; ++i) {
        std::printf("  [%d/%d] %s ... ", i + 1, kNumCases, kCases[i].name);
        std::fflush(stdout);
        int r = run_case(kCases[i], false);
        if (r != 0) {
            std::printf("FAIL\n");
            if (r == 1) ++fail_val; else ++fail_hip;
            if (nfailed < 32) failed[nfailed++] = i;
        }
    }

    std::printf("\n--- pass 1 (default stack): %d ok, %d wrong-value, %d HIP-error ---\n",
                kNumCases - fail_val - fail_hip, fail_val, fail_hip);

    if (nfailed == 0) {
        std::printf("\nRESULT: PASS -- all %d reflected arms correct on device at the\n"
                    "        DEFAULT %zu B stack. The de-recursion fix holds at runtime.\n",
                    kNumCases, stack);
        return 0;
    }

    // Diagnostic pass. If the same cases pass once the stack is raised, the
    // failure IS the unprovisioned-recursive-stack defect and not something
    // else -- that is exactly the knob the upstream report used as proof.
    std::printf("\n--- pass 2: retrying ONLY the %d failures at a 64 KB stack ---\n", nfailed);
    if (hipDeviceSetLimit(hipLimitStackSize, 65536) != hipSuccess) {
        std::printf("    could not raise the limit; skipping the diagnostic\n");
        return 1;
    }
    hipDeviceGetLimit(&stack, hipLimitStackSize);
    std::printf("    stack limit now %zu B\n", stack);

    int recovered = 0;
    for (int j = 0; j < nfailed; ++j) {
        std::printf("  [%d/%d] %s ... ", j + 1, nfailed, kCases[failed[j]].name);
        std::fflush(stdout);
        int r = run_case(kCases[failed[j]], true);
        if (r == 0) ++recovered;
        else        std::printf("still FAIL\n");
    }

    std::printf("\nRESULT: FAIL.\n");
    if (recovered == nfailed)
        std::printf("        All %d recovered at a raised stack -> this IS the\n"
                    "        unprovisioned recursive-stack defect, still live.\n", recovered);
    else
        std::printf("        %d of %d recovered at a raised stack -> at least one failure\n"
                    "        is NOT the stack defect and needs separate diagnosis.\n",
                    recovered, nfailed);
    return 1;
}
