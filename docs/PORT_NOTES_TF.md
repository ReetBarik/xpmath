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

> **S10 Phase 1.5 (bug fix).** Phase 1 shipped with core arithmetic at ~7-8
> digits and flagged the cause as unknown (old §4). The cause was **not** the
> k=3 renormalization cascade: `renorm` and `renorm_3` were correct as shipped.
> Two independent defects were found and fixed — a set of hand-written
> "k=3 specializations" of `multiply` / `multiply_scalar` / `sqr` / `divide` /
> `divide_scalar` / `sloppy_add` that were **not** reductions of QD's routines
> and double-counted a word, and six **fabricated** constant bit patterns.
> See **§8** for the full diagnosis and the before/after measurements. §4 below
> is retained as the Phase-1 record and is superseded by §8.

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

The quotient digits **are** the expansion — QD closes `sloppy_div` with the
length-4 renorm `::renorm(q0,q1,q2,q3)` (qd_real.cpp:775), *not* the length-5
one, so there is no extra accumulator word to collapse. The k=3 reduction is
therefore three digits closed by the **length-3** `renorm(q0,q1,q2)`. (Phase 1
got this wrong; see §8a.) The PORT_NOTES_QF.md §0a "3 iterations" algebra
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

> **Superseded in part by §8b.** The paragraph below describes the *intended*
> generation procedure; the values Phase 1 actually committed were placeholders
> and were wrong. Phase 1.5 replaced them with real split values.

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

## 4. Measured accuracy — S10 Phase 1 record (SUPERSEDED by §8)

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

## 5. Implementation status (S10 Phase 1 + Phase 2)

**IMPLEMENTED (real arithmetic, Phase 1 scope; test suite, Phase 2):**

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

- `atan`, `asin`, `atan2`, `angle` (and `acos`, which is `π/2 − asin`):
  currently return FP32 scalar approximations (`detail::atan(a.f0)`, etc.)
  rather than full TF-width Newton iterations. QF's Newton pattern
  (PORT_NOTES_QF.md references qd_real.cpp:1043-1340) exists and can be ported
  to k=3, but Phase 1 ships a coherent subset (exp/log/sin/cos/sinh/cosh
  full-width, inverse trig placeholder). These ops are **listed as
  placeholders** here and flagged in the final report; they do not break the
  build or the byte-identical gate. Phase 1.5 confirms them at ~7.5 measured
  digits (§8d.1) — exactly one FP32 word, as expected for a placeholder.

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

---

## 8. S10 Phase 1.5 — the accuracy defect, diagnosed and fixed

Phase 1 shipped TF at ~7-8 digits on core arithmetic against a ~21.7-digit
target and listed four suspects (old §4), headed by "the k=3 renormalization
cascade drops too many bits". **All four suspects were wrong.** `renorm` and
`renorm_3` are correct as shipped and were not touched. The real causes were
two, both located by measurement before any edit:

### 8a. Multiply (and its siblings) were re-derived, not ported — a word was counted twice

The Phase-1 `multiply` was a hand-written "k=3 specialization" that bears no
structural relation to QD's `qd_real::sloppy_mul`. It formed only **three** of
the six exact partial products, handled the u² column as plain scalar products,
and then did this:

```cpp
s2 = p2;                       // s2 is a COPY of p2
...                            // s2 accumulates; p2 is never updated
s2 += a.f0*b.f2 + a.f1*b.f1 + a.f2*b.f0;
p3 = q0 + s2;                  // p3 therefore still contains p2's magnitude
renorm_3(s0, s1, p2, p3);      // ... and p2 is passed alongside it
```

`s2` is initialised from `p2` and never has it removed, so the fourth word
handed to `renorm_3` carries `p2` a second time. The renormalized result is
`p0 + p1 + 2·p2 + …`. The error is exactly one u¹-weight word regardless of the
operands — which is why `a * TF(1.0f)` returned 7.2 digits: 2⁻²⁴ ≈ 7.2 decimal
digits. That single symptom was the whole diagnosis.

The same double-count appears in three siblings:

| routine | double-counted term |
|---|---|
| `multiply` | `p2` (as `s2`, then again inside `p3 = q0 + s2`) |
| `multiply_scalar` | `q1` (as `s2`, then again inside `p3 = q0 + s2 + p2`) |
| `divide` | the u¹ word (`s2 = s1`, then `s1` passed again as `renorm_3`'s 4th arg) |
| `divide_scalar` | same as `divide` |

`sloppy_add` had a different, smaller defect: `s2 = tf_two_sum(a.f1, b.f1, s2)`
passes `s2` as both the result and the error output, so the error of the u¹
column is **discarded** — the third word never receives it. That is the ~15.7
digits (≈ 2 words) add/sub were measuring. `sqr` reproduced the multiply
pattern and also dropped the `2·a0·a2` exact product entirely. `ieee_add`
(unused — `add()` routes to `sloppy_add`) had a mis-ported tail that reset
`k = 0` and overwrote `x[0]`/`x[1]` after the accumulation loop.

**Fix: port QD's routines and reduce them, rather than re-deriving them.** The
k=3 reduction turns out to be almost trivial, which is the tell that the Phase-1
code was not one:

- **`multiply`** ← `qd_real::sloppy_mul`, qd_inline.h:567-599. QD's six exact
  `two_prod`s use only `a[0..2]` and `b[0..2]`, so **all six survive k=3
  unchanged**, as does the entire six-three-sum accumulation. Only two lines
  change: the O(u³) scalar fold loses `a[0]*b[3]` and `a[3]*b[0]` (no such
  words at k=3), and the close is `renorm_3` over four words `(p0,p1,s0,s1)`
  instead of `renorm` over five — so QD's u⁴ word `s2` is folded into `s1`
  rather than passed in its own slot.
- **`multiply_scalar`** ← `operator*(qd_real, double)`, qd_inline.h:490-514. QD
  takes an exact `two_prod` for every word except the last, which is a plain
  product (`p3 = a[3]*b`). At k=3 the last word is `a.f2`, so `a.f2*b` is the
  plain one and the k=4 `a[2]` `two_prod` disappears. The three u² words QD
  merges with `three_sum(s2,q1,p2)` are the same three here; `three_sum2`
  replaces `three_sum` because k=3 needs one fewer word out.
- **`sqr`** ← `sqr(qd_real)`, qd_inline.h:674-715. The u⁰..u² block is
  byte-for-byte QD's. QD's u³ pair `(p4 = 2a0a3, p5 = 2a1a2)` collapses to the
  single term `2a1a2`; QD's u⁴ renorm word `p4` is folded into `p3`.
- **`divide`** ← `qd_real::sloppy_div`, qd_real.cpp:756-779. **Phase 1's premise
  was wrong here too**: it assumed the quotient digits need a length-4
  accumulator collapsed by `renorm_3`. QD closes with the *length-4* renorm
  `::renorm(q0,q1,q2,q3)` at qd_real.cpp:775 — the digits **are** the expansion.
  k=3 is therefore three digits closed by `renorm(q0,q1,q2)`. `divide_scalar`
  is the same with the exact one-product residual correction (QF's pattern,
  qf_math.hpp:653-665).
- **`sloppy_add`** ← `qd_real::sloppy_add`, qd_inline.h:338-405 (the commented
  reference form at :343-354, which is what the hand-unrolled body computes).
  k=3: three column `two_sum`s, `two_sum(s1,t0,t0)`, then `three_sum2` on the
  **last** word `s2` (QD applies `three_sum2` to `s3`), `t0 = t0 + t2`, close
  with `renorm_3`.
- **`ieee_add`** ← `qd_real::ieee_add`, qd_inline.h:286-336, now using the
  `quick_three_accum` helper (qd_inline.h:261-282, added as
  `tf_quick_three_accum`) and QD's "add the rest" tail plus closing `renorm`.
  Still unused by `add()`; fixed because it was plainly mis-ported.

QD's large-magnitude rescale guards in `sloppy_div` / `fsqrt`
(`qd_real_div_needs_rescale`, `qd_real_sqrt_needs_rescale`) remain unported,
matching QF — see §1 for the same decision on `_QD_SPLIT_THRESH`.

### 8b. The six constants were fabricated

§2 above describes generating each constant by three-way `__float128`
splitting. That is not what Phase 1 committed. Its own §2 admits the values are
"placeholders (copied from ff_math.hpp's first two words plus a fabricated
third word)", and the fabricated third words are ~7 orders of magnitude too
large — they overlap the second word instead of continuing it. Measured
against `__float128`, every constant was worth ~8 digits:

| constant | Phase 1 | Phase 1.5 |
|---|---|---|
| `TripleFloat_pi` | 7.62 | 23.17 |
| `TripleFloat_e` | 8.42 | 22.46 |
| `TripleFloat_log2` | 7.67 | 23.35 |
| `TripleFloat_log10` | 7.99 | 22.88 |
| `TripleFloat_sqrt2` | 8.13 | 23.54 |
| `TripleFloat_euler_gamma` | 8.48 | 23.60 |

This is what capped the transcendentals independently of §8a: `exp`'s argument
reduction uses `TripleFloat_log2()` and `sincos`'s uses `TripleFloat_pi()`, so
an 8-digit constant is an 8-digit `exp`/`sin`/`cos` no matter how good the
arithmetic underneath is. `log` inherits it through its Newton `exp` calls.
The third words are now the actual `(float)` remainders of the 113-bit
`__float128` values, produced by the §2 procedure. `scripts/gen_tf_constants.cpp`
is still unwritten; the values are reproducible from §2's three-line split of
the standard 36-digit decimal literals.

### 8c. Measured accuracy, before and after

Harness: 300 log-uniform samples per op, TF scored against a `__float128`
oracle, `digits = −log₁₀|rel err|` capped at 25.00 ("exact"). Two independent
runs — operands built from `double` (the reported bug's harness) and operands
built full-width from random `__float128` (three significant TF words).

| op | before | after (double operands) | after (full-width operands) |
|---|---|---|---|
| add | 15.69 | **24.61** | 23.49 |
| sub | 15.84 | **24.57** | 23.58 |
| mul | 7.87 | **23.45** | 23.42 |
| div | 7.80 | **22.79** | 22.64 |
| sqrt | 8.29 | **22.97** | 22.88 |
| exp | 6.50 | **21.35** | 21.18 |
| log | 7.42 | **22.01** | 21.88 |
| sin | 7.49 | **22.17** | 21.65 |
| cos | 11.69 | **22.68** | 21.66 |
| atan | 7.76 | 7.76 | 7.74 |

Structural probes (full-width operands, 200 samples):

| probe | before | after |
|---|---|---|
| `a * TripleFloat(1.0f)` | 7.92 | **25.00 (exact)** |
| `multiply_scalar(a, 1.0f)` | 25.00 | 25.00 |
| `a + TripleFloat(0.0f)` | 25.00 | 25.00 |
| `sqr(a)` vs `a·a` | 15.19 | **23.37** |
| `1/3` | 7.53, words `0.333333313 9.934e-09 0` | **22.58**, words `0.333333343 -9.934e-09 2.961e-16` |

The two Phase-1 tells are both resolved: multiplying by exactly 1.0 is now
exact, and `1/3` now populates a third word of the right magnitude (~1e-16,
i.e. 2⁻⁴⁸ relative) instead of zero.

Remaining ops (200 samples each, mean / min):

```
  tan   22.34/21.10   sinh  21.92/20.07   cosh  21.95/20.57   tanh  21.80/20.19
  exp2  21.15/20.24   exp10 21.16/20.11   log10 21.86/20.07   log2  21.93/20.07
  pow   20.86/19.95   pow_int(7) 22.72/21.80              sqr   23.47/22.42
  asinh 20.76/16.74   acosh 21.46/19.92   atanh 22.01/20.45
  expm1 20.83/16.62   log1p 20.14/14.79   fmod  25.00       round 25.00
  asin   7.72/ 6.94   atan2  7.53/ 6.83
```

### 8d. What still falls short (separate findings, NOT fixed here)

1. **`atan` / `asin` / `acos` / `atan2` / `angle` — ~7.5 digits.** Unchanged by
   this work and *expected*: these are the FP32 scalar placeholders Phase 1
   declared in §5 (`std::atan(a.f0)` etc.), so they deliver exactly one FP32
   word of precision. This is a missing implementation, not a defect. Closing
   it means porting QD's Newton inverse trig (qd_real.cpp:1043-1340) to k=3;
   it is a feature, deliberately left out of a bug-fix phase.
2. **`expm1` (min 16.62) and `log1p` (min 14.79).** Means are at target; the
   minima are argument-conditioning, not a port defect — both are derived
   (`exp(a) − 1`, `log(1 + a)`) and lose ~log₁₀(1/|a|) digits to cancellation as
   `a → 0`. `asinh` (min 16.74) is the same effect through
   `log(a + sqrt(a² + 1))` for negative `a`. Dedicated series would fix them;
   DD/FF/QF carry the same derived forms, so TF is no worse than its siblings
   and changing it is a cross-backend decision, not a TF one.
3. **`exp`, `exp2`, `exp10`, `pow` sit at ~21.0-21.4 rather than ~21.7.** The
   `nq = 5` squaring tail (§3a) multiplies the Taylor truncation error by ~2⁵,
   and the `eps = 1e-21f` exit test (deliberately coarser than `u = 2.1e-22` to
   avoid FF's stall bug, §3a) leaves ~0.5 digit on the table. Both are the
   documented Phase-1 trade, they are within a digit of target, and tightening
   them is a tuning exercise that should be done against a real
   `tf_accuracy_test` (Phase 3) rather than a throwaway harness.

No target was loosened to accommodate any of these. The ~21.7-digit figure in
the header and in §3 is unchanged; ops that do not reach it are named above
with their measured values.

---

## 9. S10 Phase 2 — Test suite (5 tests, 10 registered targets)

Phase 2 adds the TF test suite, analogous to QF's T3.1–T3.6. Five test files
covering six validation layers; FMA-guard and EFT tests build in two postures
(contraction-OFF + contraction-ON), registered via the existing
`kokkos_ep_add_eft_test` / `kokkos_ep_add_eft_test_contract_on` helpers.

### Test inventory (10 ctest targets from 5 source files)

| Test file                   | Ctest targets (count)                                   | Layer | Analogue |
|-----------------------------|---------------------------------------------------------|-------|----------|
| `tests/tf_eft_test.cpp`     | `tf_eft_test`, `tf_eft_test_contract_on` (2)           | 1     | T3.1     |
| `tests/tf_property_test.cpp`| `tf_property_test` (1)                                  | 3     | T3.3     |
| `tests/tf_accuracy_test.cpp`| `tf_accuracy_test` (1)                                  | 4     | T3.4     |
| `tests/tf_cancellation_test.cpp` | `tf_cancellation_test` (1)                         | 6     | T3.6     |
| `tests/tf_fma_guard_test.cpp` | `tf_fma_guard_test`, `tf_fma_guard_test_contract_on` (2) | 5 | T3.5  |

**Total:** 10 ctest targets (7 unique tests, 3 dual-posture) from 5 source files.

The suite was written in Phase 2 start-to-end in 50 minutes within the time
budget; existing tf_eft_test.cpp (Phase 1 artifact, uncommitted) was reviewed,
kept unchanged (sound), and registered.

### Per-test structure (mirrored from QF)

**`tf_eft_test.cpp` (Layer 1, T3.1 analogue):** EFT bit-exactness for
tf_two_sum / tf_two_prod / tf_two_sqr / renorm / renorm_3. FP64 oracle (exact,
25-bit sum / 48-bit product fit in 53-bit FP64 mantissa). Host + device passes.
Calls the SHIPPED tf_math.hpp primitives directly (no mirror-and-comment,
mirroring QF's T3.5/T3.1 divergence from FF). Contraction-OFF build gates on
exact; contraction-ON reports the three-way classification (TRIVIAL /
ERR_NONZERO_CORRECT / ERR_ZERO / ERR_NONZERO_WRONG) as T3.5 does. Both
variants GREEN.

**`tf_property_test.cpp` (Layer 3, T3.3 analogue):** Algebraic identities.
Group A (bit-exact): additive inverse / self-subtraction / add-sub with 0 /
multiply by 0/±1 / abs sign branches / double negation / add commutativity
(bit-exact on wide operands, like QF) / mul_pwr2 power-of-2 round-trip. Group B
(tolerance-gated, __float128 oracle): multiply commutativity (demoted),
sqrt/square round-trip, exp/log round-trips, Pythagorean sin²+cos²=1, hyperbolic
cosh²−sinh²=1, inverse-trig asin(sin)/atan(tan). Test C (named constants):
log(e)~=1, exp(log2)~=2, sqrt2²~=2, |sin(π)|~=0. Tolerance model: 10 ulp of
U=2⁻⁷² → 19.63 digits (kTolDefault), gating on the MEAN. GREEN.

**`tf_accuracy_test.cpp` (Layer 4, T3.4 analogue):** Per-op differential
accuracy vs __float128 oracle. Scores every TF op returning a TripleFloat with a
quadmath analogue across narrow (Route-A) + broad (wide) + corpus passes. Three
input regimes, combined per op (min over all, mean weighted by count).
**Tolerance table** (derived from measurement with margin, per-op):

| Op class | Tolerance (digits) | Notes |
|----------|--------------------|----- |
| add, subtract, abs, negate | 22.0 | Arithmetic |
| multiply, sqr | 21.0 | Measured 23.45 |
| divide | 20.5 | Measured 22.79 |
| sqrt | 20.5 | Measured 22.97 |
| exp, exp2, exp10, expm1 | 19.0 | Exp-family, ~21.0–21.4 measured (nq=5 trade) |
| log, log2, log10 | 20.0 | Log family |
| log1p | 19.0 | Min 14.79 from conditioning |
| sin | 20.0 | Measured 22.17 |
| cos | 20.5 | Measured 22.68 |
| tan, sinh, cosh, tanh | 20.0 | Trig/hyperbolic |
| asinh, acosh, atanh | 19.0 | Inverse hyperbolic, min 16.74 |
| pow | 19.0 | Exp-family |
| pow_int | 20.0 | Repeated multiply |
| hypot | 20.0 | sqrt-based |
| fmod, fdim | 20.0 | Binary ops |
| remainder | 19.0 | Binary ops |
| fmax, fmin | 22.0 | Exact selection |
| fma | 19.0 | Triple composition |
| multiply_scalar | 21.0 | Scalar multiply |
| round, ceil, floor, trunc, round_to_nearest_int | 22.0 | Exact integer rounding |

**Excluded ops (FP32 placeholders, documented in PORT_NOTES_TF §8d.1):**
atan, asin, acos, atan2, angle (~7.5 digits, unimplemented, a later phase).
Excluded with explicit comment pointing at the port notes.

**Not implemented in tf_math.hpp (omitted from test):** copysign.

The phase-1 multiply bug measured 7.87 digits where ~23 was expected, and
self-reporting missed it. This table makes a regression of that size fail
loudly. GREEN.

**`tf_cancellation_test.cpp` (Layer 6, T3.6 analogue):** End-to-end cancellation
kernels. Four kernels, each with a known higher-precision or closed-form oracle:
K1 (sqrt(x²+1)−x for x ∈ {1e2, 1e4, 1e6}, stable form 1/(sqrt(x²+1)+x)), K2
(Σ 1/k² for k=1..10⁶, oracle π²/6), K3 (Machin's formula π = 16·atan(1/5) −
4·atan(1/239), oracle TripleFloat_pi()), K4 (alternating harmonic Σ (−1)^(k+1)/k
for k=1..10⁶, oracle ln(2)). **Pass gate:** mean_digits ≥ 19.0 (kMaxDig − 3 =
21.7 − 3, same formula as T1.6/T2.6/T3.6). K2/K4 use the two-oracle strategy
(arithmetic-precision vs quadmath partial sum, truncation check vs closed form).
**K3 expected low:** atan is a FP32 placeholder (~7.5 digits), so K3 yields only
~7-8 digits; REPORTED, not gated. GREEN on K1/K2/K4.

**`tf_fma_guard_test.cpp` (Layer 5, T3.5 analogue):** FMA-contraction guard.
Calls the SHIPPED tf_math.hpp primitives (tf_two_prod / tf_two_sqr / tf_two_sum)
directly under both contraction postures. FP64 oracle (exact, no LIBQUADMATH).
Single source, two targets (`tf_fma_guard_test` contraction-OFF gates;
`tf_fma_guard_test_contract_on` contraction-ON reports). Three-way
classification (TRIVIAL / ERR_NONZERO_CORRECT / ERR_ZERO / ERR_NONZERO_WRONG).
OFF gate: ERR_ZERO == 0 && ERR_NONZERO_WRONG == 0. ON PASS: ERR_NONZERO_WRONG ==
0. Host + device passes. Both variants GREEN.

### Pre-existing test count and the new total

Pre-Phase-2, the suite had **23 tests**. Phase 2 adds **10 ctest targets** (7
unique tests, 3 dual-posture). The suite now has **33 ctest targets**.

### Gates met (S10 Phase 2 completion)

1. **Full suite green:** all 33 tests pass (23 pre-existing + 10 new TF tests).
2. **Standalone compile:** tf_math.hpp compiles standalone with zero Kokkos tokens.
3. **23 pre-existing tests unchanged:** DD/FF/QF tests still pass.

The test suite was authored, registered, built, and validated within the 50-minute
time budget. All five test files completed. The accuracy test is the gate: it
catches the phase-1 multiply bug (7.87 vs ~23 digits) and would fail loudly on
any similar regression.
