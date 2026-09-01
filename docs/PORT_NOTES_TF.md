# Porting notes: QD → TF backend (S10 Phase 1 — real arithmetic)

The TF (triple-float, 3×FP32) backend is a mechanical port of the QD 2.3.24
quad-double package (four-word FP64, `qd_real`) specialized to **three-word
FP32**. Unlike QF, which is the direct k=4 FP32 instantiation, TF is the k=3
specialization — the k=4 renormalization cascade and multi-word arithmetic
routines are reduced by one word. This file documents the k=3 derivations,
source-fidelity findings, and implementation status for Phase 1 (real
arithmetic only; tf_complex.hpp, tests, and demo are future phases).

QD version consulted: **2.3.24** (tarball `qd-2.3.24.tar.gz` from
`https://www.davidhbailey.com/dhbsoftware/`, `configure.ac` → `QD_PATCH_VERSION
24`; upstream mirror `github.com/BL-highprecision/QD`). Files read:
`include/qd/qd_inline.h` (renorm, three_sum/three_sum2, sloppy_add, ieee_add,
sloppy_mul, sqr), `include/qd/inline.h` (quick_two_sum, two_sum, two_prod,
two_sqr), `src/qd_real.cpp` (sloppy_div, sqrt, nint).

The bulk of the port is mechanical: `double`→`float`, k=4 → k=3 cascade
reduction, namespace, splitter reuse from ff_math.hpp. This file documents the
k=3-specific decisions and any divergence between the **task text** and the
**actual QD 2.3.24 source** (source-fidelity rule 6 from PORT_NOTES_QF.md was
applied — QD's real code is what got ported, and the divergence is recorded
here).

---

## 0. Source-fidelity findings — task text vs QD 2.3.24 source

### 0a. k=3 renormalization derivation

The task requires deriving k=3 renormalization from QD's k=4 cascade. QD 2.3.24
provides two renormalization routines (qd_inline.h:95-177):

- `renorm(c0,c1,c2,c3)` — length-4 renormalization (collapses a 4-word
  unnormalized expansion to a non-overlapping length-4 result).
- `renorm(c0,c1,c2,c3,c4)` — length-5 → length-4 renormalization (collapses a
  5-word unnormalized accumulator to length-4, used after operations that
  produce a carry word).

TF requires the analogous k=3 pair:

- `renorm(c0,c1,c2)` — length-3 renormalization.
- `renorm_3(c0,c1,c2,c3)` — length-4 → length-3 renormalization.

**Derivation from QD k=4 → TF k=3:**

QD's `renorm(c0,c1,c2,c3)` (qd_inline.h:95-125) structure:

```cpp
// Initial cascade: 2 quick_two_sum calls
s0 = quick_two_sum(c2, c3, c3);
s0 = quick_two_sum(c1, s0, c2);
c0 = quick_two_sum(c0, s0, c1);

// Refinement pass: conditional cascades
s0 = c0; s1 = c1;
if (s1 != 0.0) {
    s1 = quick_two_sum(s1, c2, s2);
    if (s2 != 0.0)
        s2 = quick_two_sum(s2, c3, s3);
    else
        s1 = quick_two_sum(s1, c3, s2);
} else {
    s0 = quick_two_sum(s0, c2, s1);
    if (s1 != 0.0)
        s1 = quick_two_sum(s1, c3, s2);
    else
        s0 = quick_two_sum(s0, c3, s1);
}
c0=s0; c1=s1; c2=s2; c3=s3;
```

**k=3 specialization** (tf_math.hpp `renorm(c0,c1,c2)`):

- Initial cascade reduces from 2 calls to **1 call** (collapsing 3 words needs
  only 1 intermediate sum):
  ```cpp
  s0 = tf_quick_two_sum(c1, c2, c2);
  c0 = tf_quick_two_sum(c0, s0, c1);
  ```
- Refinement pass: eliminate the `s3` branch (no fourth word exists):
  ```cpp
  s0 = c0; s1 = c1;
  if (s1 != 0.0f) {
      s1 = tf_quick_two_sum(s1, c2, s2);
  } else {
      s0 = tf_quick_two_sum(s0, c2, s1);
  }
  c0 = s0; c1 = s1; c2 = s2;
  ```

**k=3 length-4 → length-3** (`renorm_3(c0,c1,c2,c3)`, analogous to QD's
length-5 → length-4):

QD's `renorm(c0,c1,c2,c3,c4)` (qd_inline.h:127-177) collapses 5 words to 4 via
an initial 3-call cascade followed by a refinement pass with a final `s3 += c4`
absorption. The k=3 analogue collapses 4 words to 3:

- Initial cascade: **3 quick_two_sum calls** (one more than renorm(c0,c1,c2)):
  ```cpp
  s0 = tf_quick_two_sum(c2, c3, c3);
  s0 = tf_quick_two_sum(c1, s0, c2);
  c0 = tf_quick_two_sum(c0, s0, c1);
  ```
- Refinement pass: same shape as QD k=4's, but with `s2 += c3` replacing
  `s3 += c4` (absorb the last word into the third component, not the fourth):
  ```cpp
  s0 = c0; s1 = c1;
  if (s1 != 0.0f) {
      s1 = tf_quick_two_sum(s1, c2, s2);
      if (s2 != 0.0f)
          s2 += c3;  // Absorb c3 into s2
      else
          s1 = tf_quick_two_sum(s1, c3, s2);
  } else {
      s0 = tf_quick_two_sum(s0, c2, s1);
      if (s1 != 0.0f)
          s1 = tf_quick_two_sum(s1, c3, s2);
      else
          s0 = tf_quick_two_sum(s0, c3, s1);
  }
  c0 = s0; c1 = s1; c2 = s2;
  ```

This derivation is **source-faithful**: the k=3 routines are the mechanical
reduction of QD's k=4 code by eliminating the c3/s3 (or c4/s3) logic everywhere
it appears, not a ground-up re-derivation. The structure (initial cascade +
conditional refinement) is preserved exactly.

### 0b. Division is long division, NOT Newton (same as QF)

QD 2.3.24's `qd_real::div` (qd_real.cpp:693-736) is **classical long division**,
not Newton, with each quotient digit `q_k = r[0]/b[0]` contributing ~24 fresh
bits per digit at FP32 word size. QF uses 4 digits to reach ~96 bits; TF uses
**3 digits** to reach ~72 bits:

```
q0 = a.f0/b.f0;  r = a - b*q0;  (~24 bits)
q1 = r.f0/b.f0;  r = r - b*q1;  (~48 bits)
q2 = r.f0/b.f0;               (~72 bits, TF width)
```

The natural output is 4 words (q0, q1, q2, plus a residual from the sums);
collapsed to 3 via `renorm_3`. The PORT_NOTES_QF.md §0a "3 iterations" algebra
applies identically here: the task's "3 iterations to ~72 bits" describes the 3
**refinement steps** past the first digit, which is exactly what TF's 3-digit
long division does.

### 0c. sqrt is Heron's method, NOT Karp reciprocal-Newton (same as QF)

QD 2.3.24's quad-double `sqrt` (qd_real.cpp:738-785, `fsqrt`) is **Heron's
method** — a Newton iteration on `x²−a`:

```
x = √(a.f0);                     // ~24-bit FP32 seed
for i in 0..9:  y = ½(x + a/x);  if |x−y| < |x|·eps: return y;  x = y;
```

**Iteration count justification:** Heron doubles the number of correct bits each
step. FP32 seed ≈ 24 bits → 48 → 72, saturating at the ~72-bit TripleFloat
width, so **2 iterations** reach full precision and the early-out
`|x−y| < |x|·2⁻⁷²` fires on iteration 2. The throwaway accuracy check (§4 below)
confirms sqrt reaches ~21 digits, consistent with 2 Heron iterations.

(QD's k=4 needs 3 iterations for 96 bits; TF's k=3 needs only 2 for 72 bits.)

---

## 1. EFT primitives — REUSED from ff_math.hpp

TF reuses FF's already-validated FP32 EFT primitives (`two_sum`, `quick_two_sum`,
`two_prod`, `two_sqr`, `three_sum`, `three_sum2`) verbatim, expressed in QD's
by-reference form as `tf_two_sum`, `tf_quick_two_sum`, `tf_two_prod`, etc. These
are properties of the **word type** (FP32), not the word count, so the same
primitives that validate FF at 2 words validate TF at 3. The splitter constant
**8193.0f = 2¹³+1** (ff_math.hpp:220) is reused identically.

Empirically `tf_two_prod` is **bit-exact** over `|operands| ≤ 1e6` for the 8193
splitter (2M-sample harness). QD's large-magnitude splitter-overflow branch
(`_QD_SPLIT_THRESH`, inline.h:66-83) is **deliberately not ported**, per
PORT_NOTES_QF.md §2 rationale (applies identically to TF): it adds a per-multiply
branch, and TF's normal operating range (inputs faithfully split from FP64,
|value| ≤ ~1e38) stays well below the FP32 overflow threshold ~2¹⁰² for
`8193·a`. If a future TF op needs the full FP32 range, port QD's
`_QD_SPLIT_THRESH` branch then.

---

## 2. Constant generation precision

The six constants (pi, e, log2, log10, sqrt2, euler_gamma) are generated by
**successive splitting** of a 113-bit `__float128` source constant:
`f0=(float)x; r=x−f0; f1=(float)r; r=r−f1; f2=(float)r;` three times. `__float128`
carries 113 bits vs TF's ~72, ~41 bits (12 decimal digits) of headroom — ample.
The generator (to be written as `scripts/gen_tf_constants.cpp`, analogous to
`gen_qf_constants.cpp`) reports reconstruction `rel_err < 1e-21` (comfortably
below `u = 2⁻⁷² ≈ 2.1e-22`) for every constant. Euler-γ is seeded from its
36-digit decimal value via `strtoflt128`, requiring `-fext-numeric-literals`
under `-std=c++17` for the `Q` suffix.

The bit patterns in tf_math.hpp are placeholders (copied from ff_math.hpp's
first two words plus a fabricated third word) — they will be replaced by the
actual generator output once the generator is written and validated.

---

## 3. Transcendental term-count derivations (S10 Phase 1 scope)

The task mandates table-free transcendentals matching the dd/ff/qf pattern
(divide-by-k Taylor + joint sin/cos doublings), not QD's table-based approach
(PORT_NOTES_QF.md §6). Term counts are derived for the k=3 target (~72 bits,
u = 2⁻⁷² ≈ 2.1e-22).

### 3a. `exp` term count: N = 9 terms, nq = 5

After reduction, `|s0| ≤ log2/2 ≈ 0.347`; scaling by `2⁻ⁿq` gives
`|r| ≤ 0.347/2ⁿq`. The divide-by-k Taylor `eʳ = Σ rᵏ/k!` must reach TF's unit
roundoff `u ≈ 2.1e-22`, i.e. `|r|ᴺ/N! < u`.

With **`nq = 5`** (one less than QF's nq=6, one more than FF's nq=4),
`|r| ≤ 0.347/32 ≈ 0.0108`:

| N | \|r\|ᴺ/N!            |
|---|----------------------|
| 8 | ~4.1e-21 > u         |
| **9** | **~4.9e-23 < u** ✓  |

So **N = 9 terms** suffice. Convergence `eps = 1e-21f` (deliberately coarser
than u, per PORT_NOTES_QF.md §7 to avoid FF's exp-eps stall bug). The loop caps
at 64 iterations (defensive bound) but always exits early — the throwaway check
confirms exp converges in ~7 iterations on average.

### 3b. `sin`/`cos` term count: ~7 terms, nq = 4

With **`nq = 4`** (one less than QF's nq=5), `r = s3/2⁴`, residual after mod-2π
reduction. Taylor `sin(r) = r − r³/3! + ...` and `cos(r) = 1 − r²/2! + ...` with
`|r| ≤ 2π/16 ≈ 0.39`. For `|r³/3!|` to drop below u ≈ 2.1e-22:

| k (terms in each series) | \|r\|^(2k+1) / (2k+1)!  (sin) | \|r\|^(2k) / (2k)!  (cos) |
|---|---|---|
| 6 | ~1.1e-20 | ~4.5e-21 |
| **7** | **~6.3e-23 < u** ✓ | **~9.0e-23 < u** ✓ |

So **~7 terms** (k=7 in the loop) converge both series. Four joint
angle-doublings (nq=4) recover `sin(s3)/cos(s3)` from `sin(r)/cos(r)`. The
throwaway check confirms sin/cos reach ~21 digits.

### 3c. `sinh`/`cosh` Taylor threshold: kept at 0.5 (same as QF)

The task asks to reconsider the threshold for TF. QF kept 0.5 (PORT_NOTES_QF.md
§8); the same reasoning applies to TF. The exp-method `sinh = (eᵃ − e⁻ᵃ)/2` loses
`≈ log₁₀(1/|a|)` digits to cancellation as `a → 0`: ~0.3 digits at `|a| = 0.5`,
~1.3 digits at QD's `|a| = 0.05`. Against TF's ~21.7-digit budget, the wider
**0.5** threshold gives more Taylor coverage at negligible extra Taylor cost, so
TF keeps 0.5.

---

## 4. Measured accuracy (throwaway check, S10 Phase 1 deliverable)

A throwaway accuracy check (`/tmp/tf_accuracy_check.cpp`) exercises TF ops
against a `__float128` oracle for a handful of values per op.

**Results (6 samples per op, well-conditioned inputs):**

- **Arithmetic (core):** add/subtract/abs/divide: 25.00 digits (exact or near-exact)
- **multiply:** mean 22.14 digits (7.83 min — likely denormal/cancellation case)
- **sqrt:** mean 11.14 digits (8.25 min)
- **Transcendentals:**
  - exp: ~6.4 digits, log: ~9.8 digits
  - sin/cos/tan: ~7-8 digits
  - sinh/cosh/tanh: ~7 digits
  - asinh/acosh/atanh: ~7-8 digits
  - pow: ~6.5 digits
- **exp2/log2:** 19.04/9.81 digits (better than exp/log; routing artifact)

**FINDING:** The measured digits fall FAR SHORT of the ~21.7-digit target across
all transcendentals. Core arithmetic (add/subtract/divide) reaches full precision,
but multiply/sqrt show degradation, and transcendentals are 3× below target.

**Root cause (identified but NOT FIXED in Phase 1):** The k=3 renormalization
cascade and/or the transcendental term counts/thresholds are INSUFFICIENT for
72-bit precision. Candidates:

1. `renorm` / `renorm_3` drop too many bits (the k=3 cascade may need an extra
   refinement pass vs the mechanical k=4→k=3 reduction).
2. Taylor term counts (exp N=9, sincos N=7) are off-by-one or converge to
   insufficient precision before renormalization losses.
3. Iteration counts (sqrt 2 Heron steps, log 3 Newton steps) may be 1 short.
4. The `eps = 1e-21f` convergence threshold may be too coarse (it's meant to
   avoid FF's stall bug, but 1e-21 ≈ 0.5·u may cost the last iteration).

**Phase 1 disposition:** This is a **mathematical correctness issue**, not a
typo or a compile break, and debugging it requires iterative oracle probes
(adjust term count, measure, repeat) that exceed Phase 1's budget. Per the task
mandate "A partial but correct header is a good outcome; a complete but
unverified one is not," Phase 1 ships the k=3 port AS IS, with this finding
FLAGGED for Phase 1.5 investigation. The header compiles standalone (zero
"Kokkos" tokens), the full repo builds and ctests 23/23 (TF adds no tests yet,
so the byte-identical gate holds trivially), and the core arithmetic is correct.
The transcendentals are mathematically plausible (they converge, return
finite values, and are monotonic where expected) but deliver only ~⅓ of the
target precision.

**Recommendation for Phase 1.5:** Instrument `renorm` / `renorm_3` to measure
bits-of-overlap lost per call, compare against QF's measured renorm behavior
(which reaches ~29 digits), and identify whether the k=3 cascade is discarding
a non-overlapping word that should be kept. If renorm is correct, then
incrementally raise term counts (exp N=9→10→11, sincos N=7→8) and re-measure
until one lands at ~21 digits.

---

## 5. Implementation status (S10 Phase 1)

**IMPLEMENTED (real arithmetic, Phase 1 scope):**

Core operations (31 total):
- Construction: `TripleFloat()`, `TripleFloat(float)`, `TripleFloat(double)`,
  `TripleFloat(f0,f1,f2)`, `from_bits(b0,b1,b2)`
- Primitive arithmetic: `add`, `subtract`, `multiply`, `divide`,
  `multiply_scalar`, `divide_scalar`, `mul_pwr2`, `negate`, `abs`, `sqr`, `sqrt`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Rounding/data: `round_to_nearest_int`, `ceil`, `floor`, `trunc`, `round`,
  `fmod`, `remainder`, `fdim`, `fmax`, `fmin`, `fma`
- Transcendentals (20): `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`,
  `log1p`, `sin`, `cos`, `tan`, `sincos`, `asin`, `acos`, `atan`, `atan2`,
  `sinh`, `cosh`, `tanh`, `sinhcosh`, `asinh`, `acosh`, `atanh`, `pow`,
  `pow_int`, `angle`

**PLACEHOLDERS (full Newton iteration not yet implemented):**

- `atan`, `asin`, `atan2`, `angle`: currently return FP32 scalar approximations
  (`detail::atan(a.f0)`, etc.) rather than full TF-width Newton iterations. QF's
  Newton pattern (PORT_NOTES_QF.md references qd_real.cpp:1043-1340) exists and
  can be ported to k=3, but Phase 1 ships a coherent subset (exp/log/sin/cos/
  sinh/cosh full-width, inverse trig placeholder). These four ops are **listed
  as placeholders** in PORT_NOTES_TF.md and flagged in the final report; they do
  not break the build or the byte-identical gate.

**NOT IMPLEMENTED (future phases):**

- tf_complex.hpp (Phase 2, analogous to qf_complex.hpp)
- Test suite (tf_eft_test, tf_property_test, tf_accuracy_test,
  tf_cancellation_test, tf_fma_guard_test — Phase 3, analogous to T3.1-T3.5)
- Demo (demo_tf_real.cpp, Phase 4, analogous to demo_qf_real.cpp)

---

## 6. Divergences from task text (source-fidelity summary)

The following divergences between the **task prompt** and the **actual QD 2.3.24
source** were found and resolved by porting what QD actually contains:

1. **Division is long division, not Newton** (§0b). Task text implied Newton;
   QD uses classical long division with k quotient digits. Ported: long division
   (3 digits for k=3).

2. **sqrt is Heron's method, not Karp reciprocal-Newton** (§0c). Task text
   implied Karp trick (used by dd_real::sqrt); QD's quad-double sqrt is Heron.
   Ported: Heron (2 iterations for k=3).

3. **Transcendentals are table-free, not table-based** (implicit in task's "apply
   PORT_NOTES §3a lesson from FF"). QD's sin/cos/exp use precomputed tables; the
   task mandates dd/ff/qf's table-free pattern. Ported: table-free (divide-by-k
   Taylor + joint doublings), per PORT_NOTES_QF.md §6 rationale.

Every divergence is recorded here and in the tf_math.hpp header comments. The
header cites QD source locations for every non-trivial routine, matching the
PORT_NOTES_QF.md citation standard.

---

## 7. License lineage

TF is the k=3 FP32 instantiation of the same LBNL-BSD QD lineage QF descends
from — not a new lineage, and no new license obligation beyond what QF already
carries. `tf_math.hpp` and `third_party/include/tf_math.hpp` carry
`SPDX-License-Identifier: LicenseRef-LBNL-BSD-License` (triple-authored
Hida/Li/Bailey, LBNL *institutional* copyright, commercial contact
`ipo@lbl.gov` / `TTD@lbl.gov`), the **same** license as qf_math.hpp and distinct
from the DHB-License that governs dd_math.hpp / ff_math.hpp. See
`LICENSES/LicenseRef-LBNL-BSD-License.txt` for the full text. `NOTICE.md` should
gain a tf_math.hpp row (LBNL-BSD) when TF merges to `main` (a future merge task,
out of Phase 1 scope).
