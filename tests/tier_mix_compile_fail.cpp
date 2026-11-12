// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
//
// Compile-only probe for validation/check_tier_mix.sh (ctest tier_mix_compile_fail).
// CASE 0 is the positive control: explicit tier conversions and same-tier
// arithmetic must compile. Every other CASE mixes the strict and fast tiers
// implicitly and must FAIL to compile, for all four backends, real and complex.
// Never linked or run.
#include <xp/dd_complex.hpp>
#include <xp/ff_complex.hpp>
#include <xp/qf_complex.hpp>
#include <xp/tf_complex.hpp>

#ifndef CASE
#define CASE 0
#endif

namespace {

template <class R, class RF, class C, class CF>
void probe(R r, RF f, C c, CF cf) {
#if CASE == 0
    RF a = xp::to_fast(r);
    R b = xp::to_strict(f);
    (void)(a + f);
    (void)(b + r);
    CF z = xp::to_fast(c);
    C w = xp::to_strict(cf);
    (void)(z * cf);
    (void)(w * c);
    (void)R(f);
    (void)RF(r);
    (void)C(cf);
    (void)CF(c);
    (void)(f == a);
    (void)exp(f);
    (void)exp(cf);
#elif CASE == 1
    (void)(r + f);              // strict + fast
#elif CASE == 2
    (void)(f * r);              // fast * strict
#elif CASE == 3
    R x = f;                    // fast -> strict, implicit
    (void)x;
#elif CASE == 4
    RF x = r;                   // strict -> fast, implicit
    (void)x;
#elif CASE == 5
    R x = exp(f);               // fast function result into a strict variable
    (void)x;
#elif CASE == 6
    (void)(c + cf);             // complex strict + fast
#elif CASE == 7
    CF x = c;                   // complex strict -> fast, implicit
    (void)x;
#elif CASE == 8
    (void)(f == r);             // compare across tiers
#elif CASE == 9
    (void)(cf * r);             // fast complex * strict real
#elif CASE == 10
    (void)(c / f);              // strict complex / fast real
#else
#error unknown CASE
#endif
}

template <class R, class RF, class C, class CF>
void run() { probe(R(), RF(), C(), CF()); }

} // namespace

int main() {
    run<xp::dd_real, xp::dd_real_fast, xp::dd_complex, xp::dd_complex_fast>();
    run<xp::ff_real, xp::ff_real_fast, xp::ff_complex, xp::ff_complex_fast>();
    run<xp::tf_real, xp::tf_real_fast, xp::tf_complex, xp::tf_complex_fast>();
    run<xp::qf_real, xp::qf_real_fast, xp::qf_complex, xp::qf_complex_fast>();
    return 0;
}
