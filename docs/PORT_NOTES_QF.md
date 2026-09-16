# Porting notes: QD → QF backend (T3.0a — arithmetic + renormalization)

The QF (quad-float, 4×FP32) backend on branch `qffunKokkos` is a mechanical
port of the QD 2.3.24 quad-double package (four-word FP64, `qd_real`) down to
four-word FP32. Unlike the DD→FF port (`PORT_NOTES.md`), which translated
DDFUN, QF descends from **QD** (Hida-Li-Bailey), a different source tree with a
different license — see the "License lineage" note below.

QD version consulted: **2.3.24** (tarball `qd-2.3.24.tar.gz` from
`https://www.davidhbailey.com/dhbsoftware/`, `configure.ac` → `QD_PATCH_VERSION
24`; upstream mirror `github.com/BL-highprecision/QD`). Files read cover to
cover: `include/qd/qd_inline.h` (renorm, three_sum/three_sum2, sloppy_add,
ieee_add, sloppy_mul, sqr, operator*), `include/qd/inline.h` (quick_two_sum,
two_sum, two_prod, two_sqr, nint), `src/qd_real.cpp` (sloppy_div, accurate_div,
fsqrt/sqrt, nint(qd_real), pow(qd_real,int)).

The bulk of the port is mechanical: `double`→`float`, `x[i]`→`f{i}`, namespace,
splitter constant. This file documents the FP32-specific decisions and the two
places where the T3.0a **task text** and the **actual QD 2.3.24 source** differ
(source-fidelity rule 6 was applied — QD's real code is what got ported, and the
divergence is recorded here and in the T3.0a report).

---

## 0. Source-fidelity findings — task text vs QD 2.3.24 source

These are not bugs; they are cases where the task's algorithm sketch predates or
mis-attributes the QD 2.3.24 routine. Per the "QD-source-fidelity is
non-negotiable / do not hallucinate QD internals" mandate, the header ports what
QD 2.3.24 actually contains and cites it.

### 0a. `divide` is long division, NOT Newton

The task text describes divide as *"Newton iteration, initial reciprocal from
FP32 division, 3 iterations to ~96 bits."* QD 2.3.24's `qd_real::div`
(`qd_real.cpp:693-736`) is **classical long division**, not Newton:

```
q0 = a[0]/b[0];  r = a - b*q0;
q1 = r[0]/b[0];  r = r - b*q1;
q2 = r[0]/b[0];  r = r - b*q2;
q3 = r[0]/b[0];               // sloppy_div: 4 digits + renorm(4)
                              // accurate_div: + q4 = r[0]/b[0]; renorm_4(5)
```

Each quotient digit `q_k = r[0]/b[0]` contributes ~24 fresh bits
(q0≈24, q1≈48, q2≈72, q3≈96) and the residual `r` is refined by a full
QuadFloat multiply-subtract between digits. Four digits reach the ~96-bit
QuadFloat width. The port carries both `divide` (sloppy, the QD **default** —
QD builds with `QD_SLOPPY_DIV` on) and `divide_accurate` (5 digits + `renorm_4`).

The "3 Newton iterations, each doubling precision" reasoning in the task IS
correct arithmetic — it just describes `dd_real::sqrt`'s Karp trick, not QD's
quad-double divide. It is preserved as the sqrt-iteration justification (§0b).

### 0b. `sqrt` is Heron's method, NOT Karp reciprocal-Newton

The task text describes sqrt as *"Newton iteration, initial reciprocal from FP32
division, same posture as divide."* That is the **Karp trick** used by
`dd_real::sqrt` (`dd_real.cpp:47-72`: `x = 1/√a[0]; return a·x + …`). QD
2.3.24's **quad-double** `sqrt` (`qd_real.cpp:738-785`, `fsqrt`) is instead
**Heron's method** — a Newton iteration on `x²−a`:

```
x = √(a[0]);                          // ~24-bit FP32 seed
for i in 0..9:  y = ½(x + a/x);  if |x−y| < |x|·eps: return y;  x = y;
```

The port keeps QD's exact structure (up-to-10-iteration loop with early-out,
`eps = 2⁻⁹⁶`). **Iteration count justification** (confirming the task's "3"):
Heron doubles the number of correct digits each step. FP32 seed ≈ 24 bits →
48 → 96, saturating at the ~96-bit QuadFloat width, so **3 iterations** reach
full precision and the early-out `|x−y| < |x|·2⁻⁹⁶` fires on the 3rd. (The
smoke test confirms sqrt mean = 29.0 digits.) Note Heron needs a full QuadFloat
`divide` per step, so QF sqrt is heavier than a reciprocal-Newton would be —
this is inherited from QD's choice, not a port decision.

### 0c. Newton-iteration-count algebra (task deliverable)

Task asks to confirm 3 iterations for both divide and sqrt at FP32:

- **sqrt (Heron, §0b):** 24 → 48 → 96 bits ⇒ **3** iterations. ✓ (matches
  QD's early-out behavior).
- **divide:** QD is not Newton (§0a), but the *equivalent* precision growth
  holds per **quotient digit**: 24 → 48 → 72 → 96 bits ⇒ **4 digits** for
  sloppy_div (the default), **5** for accurate_div. So "3 doublings from a
  24-bit seed" = 3 *refinement* steps past the first digit, which is what QD's
  4-digit long division does. The task's "3 iterations" and QD's "4 digits"
  describe the same 96-bit target; the count differs only because long division
  counts the seed digit separately.

---

## 1. Dekker splitter reuse (FP32)

QF reuses FF's already-validated FP32 Dekker two_product (`two_prod`,
`ff_math.hpp:266-274`) and its splitter `8193.0f = 2¹³+1` — expressed in QD's
by-reference form as `qf_two_prod` (cites QD `inline.h:85-99`). Empirically
`two_prod` is **bit-exact** over `|operands| ≤ 1e6` for both `4097` and `8193`
(2M-sample harness); `8193` is chosen for FF parity. The FP32 EFT primitives
(`qf_two_sum`, `qf_quick_two_sum`, `qf_two_sqr`, `qf_three_sum`,
`qf_three_sum2`) are bit-identical to the transforms embedded in FF's
`add`/`multiply`, per the "reuse FF's EFT primitives, do not re-derive" mandate.

## 2. Splitter-overflow limit NOT ported (QD `_QD_SPLIT_THRESH`)

QD's `split()` (`inline.h:66-83`) has a large-magnitude branch that pre-scales
by `2⁻²⁸` when `|a| > _QD_SPLIT_THRESH = 2⁹⁹⁶` to avoid splitter overflow in
FP64. The FP32 analogue threshold is ~`2¹⁰²` (where `8193·a` overflows FP32's
~3.4e38 = 2¹²⁸ range). This branch is **deliberately not ported** in T3.0a: it
adds a per-multiply branch to every partial product, and QF's normal operating
range (inputs faithfully split from FP64, |value| ≤ ~1e38) stays well below the
threshold. The consequence surfaced once in the smoke test — a directed
`huge/tiny` divide whose quotient is `1e36` (> 2¹⁰²·⅟8193) drove `8193·q` to
`inf`→`NaN` inside `two_prod`, exactly the FP32 analogue of PORT_NOTES §4a
(FF `exp` splitter overflow at input > 79). It is guarded as an
out-of-safe-domain input in the smoke test (skip-not-fail, matching T1.1's
splitter-overflow guard), NOT worked around in the arithmetic. If a later T3.x
op needs the full FP32 range, port QD's `_QD_SPLIT_THRESH` branch then.

## 3. `sloppy_add` safety at the narrower FP32 exponent

QD's default add is `sloppy_add` (`qd_inline.h:338-405`; QD builds with
`QD_IEEE_ADD` **off**), which forgoes the digit-by-digit `quick_three_accum`
merge of `ieee_add` in favor of a fixed 4-wide component-wise `two_sum` chain.
Its correctness rests on the operands being **non-overlapping expansions**
(each `f_{i+1} ≤ ½ ulp(f_i)`), not on exponent range — the same property FP32
and FP64 both provide. So the DD→FF concern about FP32's 6×-narrower exponent
(PORT_NOTES §4a) does **not** make sloppy_add unsafe: no splitter, no
magic-constant, no dynamic-range assumption is involved. The port therefore
keeps QD's default `sloppy_add` (with `ieee_add` also provided for parity and
tight-bound callers). The smoke test confirms add/subtract at mean 29.0 digits
with no observed catastrophic loss. **If** a future accuracy sweep (T3.4) shows
sloppy_add failing a published bound at FP32's exponent extremes, switch the
`add()` dispatch to `ieee_add` and cite the failing corpus there — that switch
is a one-line change and both routines are already present.

## 4. `nint` — the FF `ffnint` bug does NOT recur

FF's `round_to_nearest_int` hit an off-by-one near half-integers because the
DD-style `2^(2p−1)` magic-constant trick is ill-conditioned at FP32's 24-bit
mantissa (PORT_NOTES §4b), and FF had to reroute rounding through FP64. QD's
`nint` (`inline.h:116-120`, `qd_real.cpp:48-86`) does **not** use the
magic-constant trick at all — it is `floor(d+0.5)` per component with
half-integer tie corrections keyed on the sign of the next component. That is
well-conditioned at every FP32 magnitude, so the FF bug has no analogue here and
`round_to_nearest_int` is a direct, unmodified port. (No FP64 detour needed.)

## 5. Constant generation precision

`scripts/attic/gen_qf_constants.cpp` extends the FF Route-A generator to four words by
**successive splitting** of a 113-bit `__float128` source constant:
`f0=(float)x; r=x−f0; f1=(float)r; …` four times. `__float128` carries 113 bits
vs QF's ~96, ~17 bits (5 decimal digits) of headroom — ample. Reconstruction
`rel_err` for every constant is `< 6e-31` (reported inline by the generator),
comfortably below `u = 2⁻⁹⁶ ≈ 1.3e-29`. Euler-γ is not a libquadmath literal, so
it is seeded from its 36-digit decimal value via `strtoflt128`. Requires
`-fext-numeric-literals` under `-std=c++17` for the `Q` suffix (documented in
the generator's build comment).

---

# Porting notes: QD → QF backend (T3.0b — transcendentals)

The T3.0b transcendental block (`exp`/`log`/`sin`/… appended to
`qf_math.hpp` after the T3.0a arithmetic block) ports QD 2.3.24
`qd/src/qd_real.cpp`. As with T3.0a, most of it is mechanical, but the
transcendentals surfaced a larger structural divergence from the *task text*
than the arithmetic did, plus several FP32-narrower decisions. All are recorded
here (rule 6 / rule 8).

## 6. Source-fidelity — QD's transcendentals are TABLE-BASED; QF is table-free

This is the T3.0b analogue of §0 (T3.0a's div/sqrt findings), and it is the
biggest single divergence in the whole QF port.

**QD 2.3.24's actual transcendentals use precomputed tables:**

- `exp` (`qd_real.cpp:925-983`): reduces `a` by `m·log2`, scales the residual by
  `inv_k = 2⁻¹⁶`, runs a Taylor loop that multiplies by a **15-entry `inv_fact[]`
  table** (`qd_real.cpp:891-923`, the reciprocal factorials `1/3!…1/17!`), then
  **squares 16 times** via `s = 2s + s²`.
- `sin`/`cos`/`sincos` (`qd_real.cpp:2136-2360`): reduce mod 2π, then mod π/2,
  then mod π/1024, and look up a **256-entry `sin_table` and 256-entry
  `cos_table`** of `sin(kπ/1024)` / `cos(kπ/1024)` (`qd_real.cpp:945-2050`),
  combining them with a short `sincos_taylor` on the |t| ≤ π/2048 residual.

**The QF port is table-free.** It uses the divide-by-k Taylor + joint-doubling
structure of the sibling `dd_math.hpp` / `ff_math.hpp` headers instead:

- `exp`: divide-by-k Taylor (`term = rᵏ/k!` by dividing by `k` each step, no
  `inv_fact` table), `nq = 6` scaling, 6 squarings.
- `sin`/`cos`/`sincos`: reduce mod 2π only, then a divide-by-k Taylor on
  `r = s3/2⁵` followed by 5 **joint** sin/cos angle-doublings (PORT_NOTES §3a),
  no π/1024 tables.

**Why deviate from QD-actual here (but not in T3.0a arithmetic)?** Three reasons,
in priority order:

1. **The T3.0b task text explicitly directs it.** It calls for "joint sin/cos
   doublings — apply PORT_NOTES §3a lesson from FF" and an `exp` "Taylor with
   argument reduction; more terms than QD's FP64 version — derive term count."
   Both descriptions are the *table-free* algorithm, not QD's table lookup.
   Porting QD's tables would contradict the task's own §3a/§4a mandate.
2. **Tables are device-hostile.** A 256-entry × 4-word `sin_table` +
   `cos_table` (2048 FP32 constants) plus the `inv_fact` table would live in
   CUDA constant/global memory or bloat every kernel's register/constant
   footprint. `dd_math.hpp` and `ff_math.hpp` are the repo's designated
   *portable, table-free* multi-word references; QF is meant to read as their
   4-word sibling, not as a different (table-based) family.
3. **QF's 4-word π makes the table unnecessary for the accuracy target.** QD
   needs the π/1024 table because its ~64-digit target cannot be reached by
   Taylor on a residual reduced only mod 2π (the residual is too large). QF's
   ~29-digit target *is* reachable: `_2pi = mul_pwr2(pi, 2)` is exact from the
   T3.0a 4-word π (~2⁻⁹⁶), so the mod-2π residual is accurate enough that a
   32-doubling (2⁵) Taylor converges in ~9 terms to full QF width. Measured:
   sin/cos mean 28.5 digits, min 26 (see demo run below).

**Consequence for T3.6 (a plus, not a regression).** PORT_NOTES.md §5 lists
"`sin`/`cos` near ±π" as *not fixable* in FF because FF's 2-word π carries only
~14 digits, so argument reduction near π loses ~6 digits. QF's 4-word π does not
have that ceiling, so near-π `sin`/`cos` are distinguishable from noise — exactly
the T3.6 adversarial goal. This is a direct benefit of the wider type, available
*because* the reduction uses the accurate 4-word `_2pi` rather than a table.

Every ported transcendental still **cites the QD 2.3.24 routine it mirrors
mathematically** and flags where it follows dd/ff structure. `sin_table`,
`cos_table`, and `inv_fact` are therefore **not** added to
`scripts/attic/gen_qf_constants.cpp` (§5 above is unchanged — only the six scalar constants are
generated). The Newton skeletons (`log`, `atan2`) ARE faithful QD ports.

## 7. `exp` term-count derivation (T3.0b deliverable)

After reduction, `|s0| ≤ log2/2 = 0.347`; scaling by `2⁻ⁿq` gives
`|r| ≤ 0.347/2ⁿq`. The divide-by-k Taylor `eʳ = Σ rᵏ/k!` must reach QF's unit
roundoff `u = 2⁻⁹⁶ ≈ 1.3e-29`, i.e. `|r|ᴺ/N! < u`.

With **`nq = 6`** (matching `dd_math.hpp`), `|r| ≤ 0.347/64 ≈ 5.4e-3`:

| N | \|r\|ᴺ/N!            |
|---|----------------------|
| 9 | ~7e-25               |
|10 | ~4e-27               |
|**11**| **~1e-30 < u** ✓  |

So **N = 11 terms** suffice. Compared to QD's FP64 `exp`: QD uses `nq = 16`
(k = 2¹⁶, so `|r| ≤ 5.3e-6`) and its `inv_fact` loop caps at `i < 9` terms for a
~64-digit target. QF needs *more Taylor terms* (11 vs 9) only because it reduces
far less aggressively (nq = 6 vs 16) — the arithmetic-per-term is cheaper (2×FP32
words fewer) and it needs no factorial table. Net: fewer squarings (6 vs 16),
slightly longer Taylor, no table.

The convergence `eps` in the loop is set to **`1e-28f`**, deliberately *coarser*
than `u`. This avoids the FF `exp`-eps bug (`ff_math.hpp` used `eps = 1e-15f`
finer than the FloatFloat resolution 3.55e-15, so `|term| ≤ eps·|sum|` could
never fire → spurious iteration-limit stalls and wrong 0-returns). The QF loop
also does **not** return 0 on the iteration cap (another latent FF behavior); it
falls through with the best partial sum. Same coarse-eps / no-return-on-cap
posture is applied to `sincos`, `expm1`, `sinhcosh`, and `atanh`.

## 8. `sinh`/`cosh` Taylor threshold: kept at 0.5, NOT shifted to QD's 0.05

The task asks to "reconsider at QF precision." QD uses a Taylor branch for
`|a| < 0.05` (`qd_real.cpp:2513`); `ff_math.hpp` uses `|a| < 0.5`. The exp-method
`sinh = (eᵃ − e⁻ᵃ)/2` loses `≈ log₁₀(1/|a|)` digits to cancellation as `a → 0`:
~0.3 digits at `|a| = 0.5`, ~1.3 digits at `|a| = 0.05`. Against QF's ~29-digit
budget, the wider **0.5** threshold gives more Taylor coverage (avoiding the
cancellation over a larger neighborhood) at negligible extra Taylor cost, so the
QF port keeps FF's 0.5 rather than QD's 0.05. `cosh = (eᵃ + e⁻ᵃ)/2` has no
cancellation and is always taken from the exponentials for `|a| ≥ 0.5`. Measured:
sinh/cosh mean 28.2 digits (demo). `asinh`/`acosh` need no Taylor branch (their
log arguments never approach 1); only `atanh` gets the `|a| < 0.5` Taylor branch
(PORT_NOTES §3c), since its `log((1+a)/(1−a))` form evaluates log near 1 for
small a.

## 9. §4b remainder sign — no FP32-specific divergence observed

The T1.2/T2.2 precedent warned that FP32 `remainder`/`drem` can sign-disagree
with the mathematical convention. QF's `remainder` is `a − b·nint(a/b)` (faithful
port of QD `drem`, `qd_real.cpp:2462`), reusing the T3.0a `round_to_nearest_int`
(QD's `floor(d+0.5)`-based `nint`, which — per T3.0a §4 — does **not** suffer the
FF `ffnint` magic-constant bug). In the T3.0b demo (`--batch 5000`, seed 12345,
inputs `a∈[0.1,100]`, `b∈[0.1,10]`) `remainder` scores **mean 29.00 / min 29.00**
digits vs the `__float128` oracle — i.e. **no** sign disagreement was observed at
FP32 for QF. `remainder` is nonetheless listed as conditioning-limited in the
demo verdict (result → 0 near a multiple of b makes relative error unbounded, per
PORT_NOTES §5), independent of any sign question. **No fix introduced**: the
routine matches `std::remainderf`'s host behavior via the shared QD `nint`. If a
later wider sweep (T3.4) finds a sign case, record the corpus there.

## 10. `exp` final scaling and denormal tail (§4a applied; §5 inherited)

`exp`'s final `× 2ⁿᶻ` scales each of the four components directly
(`s3.f0*pow2, …`), **not** through `multiply_scalar` — PORT_NOTES §4a: at large
`nz` (≥116) `multiply_scalar` would compute `8193·2ⁿᶻ` inside Dekker splitting and
overflow FP32. This is also what QD does (`ldexp(s, m)`, component-wise). The
`exp` accuracy tail (demo: mean 25.99, min 10.49 over `a∈[−80,80]`) is the
**inherited** PORT_NOTES §5 conditioning limit, not a scaling bug: for `a ≈ −80`,
`eᵃ` lands near FP32's smallest normal (~1.18e-38) and the low-order QF words fall
into FP32 denormal range, capping digits in the tail. `exp` is therefore listed
conditioning-limited in the demo verdict; its *median* (27.96) and bulk are at
full QF width.

## 11. Demo pass/fail verdict + measured accuracy (T3.0b deliverable)

`src/demo_qf_real.cpp` (adapted from `demo_ff_real.cpp`) exercises all 39 real
ops against the `__float128` oracle and returns **RC 0 iff every op meets a
mean ≥ 24.0-digit gate**, with conditioning-limited ops (exp, asin, acos, atanh,
fmod, remainder, sub, fdim, fma — grounded in PORT_NOTES §5) exempt from the fail
verdict. Run `--batch 5000 --repeats 2` (Serial/host backend): **39 pass, 0 fail,
RC 0**. Per-op mean digits: arithmetic/rounding/data ops 29.00; log-family
28.99; sin/cos 28.57, tan 28.89; asin 28.68 / acos 28.85 / atan 28.98; sinh/cosh
28.2, tanh 28.87; asinh/acosh 28.95–28.98, atanh 28.72; pow 27.77; exp 25.99
(denormal tail, §10); exp2/exp10 26.79. Independent host probe (2000 samples/op,
`/tmp/qf_accuracy_probe.cpp` pattern) agrees within 0.1 digit.

---

# Porting notes: QF complex layer (T3.0c — qf_complex.hpp)

`third_party/include/qf_complex.hpp` is a scalar-swap port of
`ff_complex.hpp` (FloatFloat → QuadFloat), which is itself a DD→FF translation
of `dd_complex.hpp`. Most of it is mechanical — the complex composition formulas
((ac−bd)+(ad+bc)i product, Kahan complex sqrt, Euler exp, polar log) are textbook
identities dispatching to `qf_math.hpp`'s LBNL-BSD QuadFloat routines. Four
complex-specific findings are recorded below (rule 6 / rule 8).

## 12. `sincos` / `sinhcosh` OUTPUT-ARGUMENT ORDER differs between ff_math and qf_math

The single biggest porting hazard in T3.0c — a silent-wrong-answer trap if copied
verbatim. The two real backends name their joint-output params in **opposite
order**:

- `ff_math.hpp`: `sincos(a, x, y)` writes `x = cos(a)`, `y = sin(a)`;
  `sinhcosh(a, x, y)` writes `x = cosh(a)`, `y = sinh(a)`
  (see the trailing `x = cos_r; y = sin_r;` at the end of each ff routine).
- `qf_math.hpp`: `sincos(a, sin_a, cos_a)` — **sin first**;
  `sinhcosh(a, sinh_a, cosh_a)` — **sinh first** (the T3.0b out-param names).

`ff_complex.hpp` calls e.g. `sincos(z.im, c, s)` relying on the FF (cos, sin)
order. Porting that call verbatim to QF would bind `c` to `sin` and `s` to `cos`
— every complex exp/sin/cos/sinh/cosh/tanh/polar would silently swap its
sin/cos components. `qf_complex.hpp` therefore passes the local (cos, sin) /
(cosh, sinh) variables in **swapped positional order** at every call site
(`sincos(z.im, s, c)`, `sinhcosh(z.im, sb, cb)`, …), so the local variable
*meanings* (`c` = cos, `s` = sin, `cb` = cosh, `sb` = sinh, …) stay identical to
`ff_complex.hpp` and the downstream algebra is byte-for-byte the same. This is
called out in the `qf_complex.hpp` header preamble and inline at each call. There
is no numeric deviation once the swap is applied — measured sin/cos/sinh/cosh
means are 28.1–28.3 digits, consistent with the real T3.0b transcendentals.

## 13. QD 2.3.24 ships NO complex header — ff_complex/dd_complex are the sole refs

The T3.0c preamble mandate is to read `qd/src/qd_complex.cpp` (or
`dd/src/dd_complex.cpp`) before porting. **Neither exists in QD 2.3.24.** A
`grep -ril complex` over the whole 2.3.24 tree matches only NEWS / README / TODO
/ Fortran files; `qd/include/qd/` and `qd/src/` carry `qd_real.{h,cpp}`,
`dd_real.{h,cpp}`, and the `c_dd`/`c_qd` C-linkage wrappers of the **real** types
only. QD's quad-double complex is left for users to layer on. Consequently
`ff_complex.hpp` + `dd_complex.hpp` are the **sole** algorithm references for
`qf_complex.hpp`; there is no QD complex routine to cite, and each function
therefore cites the `ff_complex.hpp` (and, as the deeper DDFUN-derived reference,
`dd_complex.hpp`) line range it mirrors, not a QD location.

## 14. `norm` / `arg` are additions, not ports (no ff/dd analogue)

The T3.0c op inventory asks for `abs / norm / arg` among the "standard complex
ops". `ff_complex.hpp` / `dd_complex.hpp` ship only `abs` and `conj` as standalone
basic ops; they expose the polar angle solely *inside* `log()` (via `atan2`) and
never define `norm`. So `qf_complex.hpp` adds two functions with **no upstream
line to cite**, using the `std::complex` conventions:
`norm(z) = re²+im²` (the *squared* magnitude, no sqrt — the C++ standard's
`std::norm`, not the "vector norm" = `abs`) and `arg(z) = atan2(im, re)`. Both are
flagged in-header as inventory additions rather than ports. Measured `abs` mean
29.0, and `norm`/`arg` reuse the same `multiply`/`add`/`atan2` primitives.

## 15. Complex-op conditioning list + demo verdict (T3.0c deliverable)

`src/demo_qf_complex.cpp` (adapted from `demo_ff_complex.cpp` + the T3.0b
`demo_qf_real.cpp` verdict) exercises all 24 complex ops against the
`__complex128` oracle and returns **RC 0 iff every op meets a mean ≥ 24.0-digit
gate in BOTH the real and imag components**, with conditioning-limited ops exempt.
The complex conditioning list (extends PORT_NOTES §5 to the complex layer):
`sub` (near-cancellation), `div` (small-|denominator| amplification), `tan`
(cos z → 0), `asin`/`acos`/`atan`/`atanh` (branch-point derivative blow-up),
`log`/`log10` (|z| → 1 ⇒ real part → 0), `pow` (`exp(w·log z)` compounds log's
conditioning). Run `--batch 5000 --repeats 2` (Serial/host backend): **24 pass,
0 conditioning-exempt, 0 fail, RC 0** — every op cleared the gate on its own, so
none needed the exemption. Per-op mean digits (real / imag): add/sub 29.00/29.00;
mul 28.95/29.00; div 28.99/28.94; abs 29.00; conj 29.00; sqrt 28.99/28.99; exp
28.12/28.12; log 28.45/28.70; log10 28.45/28.70; sin 28.28/28.17; cos 28.28/28.17;
tan 28.08/28.70; asin 28.64/27.82; acos 28.88/27.82; atan 28.92/26.81;
sinh 28.15/28.19; cosh 28.19/28.14; tanh 28.37/27.70; asinh 28.36/28.40;
acosh 28.51/28.68; atanh 27.93/28.78; pow 27.78/27.81; polar 28.56/28.57. All
well-conditioned ops sit at the ~28–29-digit QF ceiling; the lowest means
(atan-imag 26.81, tanh-imag 27.70, pow 27.8) are the branch-cut / compounded
cases flagged above, still above the 24-digit gate.

---

# Porting notes: QF normalization form (T3.2 — non-overlap gate posture)

## 16. QD renorm delivers Shewchuk-weak non-overlap, NOT strict Priest (T3.2 posture)

`tests/qf_nonoverlap_test.cpp` (T3.2) checks the length-4 non-overlap invariant of
every QF op. Its original posture was the **strict Priest** bound
`|f_{i+1}| <= 1/2 ulp(f_i)`. The test surfaced a systematic, ~1e-6-rate deviation
across nine ops that lands a word in the half-open band `(1/2 ulp, ulp]`. This
section records the algorithmic finding that grounds T3.2's decision to adopt the
**Shewchuk-weak** gate (`kStrictPriestGate = false`), not a code fix.

**(a) Algorithmic finding — renorm is a `quick_two_sum` cascade.** QD 2.3.24's
normalization (`renorm(c0,c1,c2,c3)` and the length-5 `renorm(c0..c4)` used after
carry-producing ops, both Hida-Li-Bailey, `qd/include/qd/qd_inline.h`) is built
**entirely** from `qd::quick_two_sum` — verified directly from the QD source:
`renorm` chains `s0 = quick_two_sum(c2,c3,c3); s0 = quick_two_sum(c1,s0,c2);
c0 = quick_two_sum(c0,s0,c1); …`. `quick_two_sum(a,b)` (a.k.a. Dekker's
`Fast2Sum`) requires the precondition `|a| >= |b|` and returns `s = fl(a+b)` with
the exact error `e = fl(b-(s-a))`, so `a+b = s+e` holds *exactly* (Dekker 1971).
For a **single** pair that error is `|e| <= 1/2 ulp(s)` — Priest-strong. But a
renormalizing *cascade* re-feeds each intermediate error into the next
`quick_two_sum`, and the running low-order word absorbing those chained errors is
only guaranteed to be **non-overlapping in the weak (Shewchuk) sense**
`|f_{i+1}| <= ulp(f_i)`, never the strict Priest `1/2 ulp`. The cheap cascade
buys speed at the cost of the tighter bound; recovering strict Priest requires an
extra normalization sweep (Priest's renormalization), which QD deliberately does
not perform.

**(b) Citations.** The weak-vs-strong distinction and the guarantee of the fast
cascade are Shewchuk, *"Adaptive Precision Floating-Point Arithmetic and Fast
Robust Geometric Predicates"*, Discrete & Comput. Geom. 18:305–363 (1997) — which
defines *nonoverlapping* expansions in the weak `<= ulp` sense and proves the fast
renormalization preserves exactly that — versus Priest, *"Algorithms for Arbitrary
Precision Floating Point Arithmetic"* (1991), whose stronger `<= 1/2 ulp`
non-overlap needs the costlier renorm. For the modern tight error bounds of the
underlying double-word building blocks (`Fast2Sum` / `2Sum`) on which the cascade
rests, see Joldes/Muller/Popescu, *"Tight and rigorous error bounds for basic
building blocks of double-word arithmetic"*, ACM TOMS 44:2 (2017), Theorem 2 & §4.
The algorithmic fact for QF (renorm = `quick_two_sum` cascade ⇒ weak bound) is
independently confirmed from the QD source cited in (a), which is the load-bearing
evidence here; the length-4 expansion argument itself is Shewchuk/Priest (a 4-word
expansion, beyond the double-word scope of the TOMS paper).

**(c) Why this differs from §5 (and §10/§11/§15).** §5 ("Constant generation
precision") and the *conditioning family* that cites it (§10 exp denormal tail,
§11 real demo verdict, §15 complex-op conditioning list) concern **accuracy** —
how many correct digits an op delivers *given its inputs*, limited by input
conditioning, cancellation, and the FP32 denormal floor. §16 is a distinct
category: **normalization form** — the bit-overlap *structure* of a
correctly-normalized result, independent of how accurate that result is. A word in
`(1/2 ulp, ulp]` is not a lost-digits problem (the value is represented to full QF
width); it is a statement about the packing shape of the four words. Conflating
the two would mis-file a normalization-form clarification as an accuracy defect, so
they are kept as separate registries. This is a load-bearing distinction: real
precision loss is assessed in T3.4 (accuracy vs quadmath), not by this invariant.

**(d) Why NOT a B-task.** There is no library defect to fix. The weak bound is a
*deliberate design choice* in QD's fast renormalization, inherent to the
`quick_two_sum` cascade; tightening it to strict Priest would require replacing
`renorm`/`renorm_4` with a Priest-style renormalization — a change to `qf_math.hpp`
barred by Rule 4 (a validation task reports and stops; it does not patch the
library). qf_math.hpp is a *faithful QD port*, and a faithful port inherits QD's
normalization form. No B-task is warranted.

**(e) Test posture adopted.** `kStrictPriestGate = false` (T3.2 commit `353193d`):
NOVL_WEAK deviations (`1/2 ulp < |f_{i+1}| <= ulp`) are tolerated by the gate but
**still counted per-op with worst-ratio reported** in the test log, so any drift is
visible. NOVL_FAIL — a word `> ulp(f_i)`, a packing break (nonzero word after a
zero), or a NaN/inf leak in a checkable slot — remains **always fatal**; that is
what would signal genuine corruption. The strict 1/2-ulp check still runs on every
output; only whether WEAK is fatal changed.

**(f) Measured extent (strict-Priest diagnostic run, kRandomN = 2×10⁵).** 19 WEAK
deviations across 9 ops — log:1, cos:1, sincos.cos:2, subtract:1, fmod:4,
remainder:7, fdim:1, multiply_scalar:1, pow_int:1 — over 11,234,222 checked
results (72,362 skipped as out-of-domain). Worst overlap ratio **1.375×** (fmod;
1.0 = exactly 1/2 ulp, 2.0 = exactly ulp), i.e. every deviation stays strictly
below `ulp`. **Zero NOVL_FAIL** anywhere; device tripwire (5 ops) and all 13 corner
cases clean. fmod/remainder — the argument-reduction-heavy ops — dominate, as
expected for a renorm-bound artifact.

**(g) Downstream implication.** Every QF op cascade already relies only on
`quick_two_sum`'s `|a| >= |b|` precondition (which the weak form preserves), not on
½-ulp separation, so weak non-overlap is sufficient for correct downstream
arithmetic; `qf_complex.hpp` inherits the same form through the scalar dispatch. The
real question of *how many digits* QF delivers is orthogonal to this and is settled
by T3.4, not by the non-overlap gate.

## 17. `divide_scalar` — QF's missing scalar-divide primitive

DD and FF have carried a `divide_scalar(T, scalar)` since the DDFUN port. QF
did not. Every Taylor loop in `exp`, `expm1`, `sincos`, `sinhcosh`, and `atanh`
divides by a small loop integer each iteration, and with no scalar form those
seven call sites promoted the integer to a full `QuadFloat` and paid
`divide()`'s four-word long division.

Cost of the gap, by static op count (`README.md` Section 2 appendix
convention): `divide()` is 643 native FP32 ops — four quotient digits, each
residual corrected by a full `multiply_scalar` (115) plus `subtract` (92). The
divisor being a single float makes that correction ONE exact product:
`qf_two_prod` returns it as a non-overlapping `(p, e)` pair, so `(p, e, 0, 0)`
is a valid QuadFloat and the correction costs 17 + 92 = 109 instead of 207.
`divide_scalar` therefore lands at 349 ops against 643 — a 46% cut at the
primitive, and the loops call it once per iteration.

Effect on the transcendentals (static counts, before -> after):

| Op | before | after | cut |
|---|---|---|---|
| `exp` | 11754 | 8814 | 25% |
| `log` | 36463 | 27643 | 24% |
| `sin` / `cos` | 20044 | 15340 | 23% |
| `pow` | 48432 | 36672 | 24% |
| `atan` | 61865 | 49517 | 20% |

Measured wall time agrees and is slightly better than the static count
predicts, which is expected — the removed work sits on `renorm`'s serial
`quick_two_sum` dependency chain, where the counts understate the benefit.
At `--batch 50000 --repeats 1`: `log` 3601 -> 2345 ms (35%), `exp` 1030 -> 756
ms (27%), `sin` 1649 -> 1257 ms (24%). The full 23-test ctest wall dropped from
320 s to 254 s.

**This is not a precision/speed trade.** The algorithm is identical to
`divide()` — same four-digit long division, same closing `renorm` — and the
residual correction is exact rather than merely faithful, so if anything the
scalar path is marginally better conditioned. Verified by re-running the full
39-op QF accuracy sweep at `--batch 1000000 --repeats 5 --seed 12345` before
and after: all 39 rows (min/max/median/mean) are **bit-identical**. `ctest`
stays 23/23.

Why QD itself does not show this gap: QD's `qd_real` transcendentals are
table-based (see §6), so they multiply by a stored `inv_fact[]` entry instead
of dividing by a loop counter at all. The table-free port adopted in §6 is what
introduces the division, and DD/FF absorbed it with `divide_scalar` while QF
was left without one.

## License lineage (complex layer — T3.0c)

`qf_complex.hpp` carries `LicenseRef-LBNL-BSD-License` — the **same** license as
`qf_math.hpp`, **not** the `LicenseRef-DHB-License` that governs its structural
template `ff_complex.hpp` / `dd_complex.hpp`. Two precedents were weighed:
**(a)** follow scalar dispatch (LBNL-BSD, as T3.0a chose for `qf_math.hpp`) —
every arithmetic call inside the header dispatches to LBNL-BSD QuadFloat routines;
**(b)** follow the structural template (DHB, as `ff_complex.hpp` inherits from
`dd_complex.hpp`'s DDFUN heritage). **Decision: (a) LBNL-BSD.** Rationale: the
file contains no DDFUN/DHB-original arithmetic — the complex composition formulas
are textbook identities, not DDFUN inventions, and every non-trivial numeric step
is a QD-derived QuadFloat op. A DHB header would attribute copyright to Bailey
personally and cite the wrong commercial contact for a file whose substance is the
LBNL institutional QD package; it also keeps the whole QF backend
(`qf_math.hpp` + `qf_complex.hpp`) under one consistent license. Reet to confirm
in review. `NOTICE.md` should gain a `qf_complex.hpp` row (LBNL-BSD) when QF
merges to `main` (a T3.x merge task, out of T3.0c scope).

---

## License lineage (T3.0a kickoff action item, TEST_SUITE_PLAN §"Phase 2/3
open question")

QF is modeled on **QD** (`qd_real.cc`), NOT DDFUN. QD carries the
**LBNL-BSD-License** (modified BSD-3-Clause + §3 grant-back), triple-authored
Hida/Li/Bailey with LBNL *institutional* copyright and commercial contact
`TTD@lbl.gov` / `ipo@lbl.gov` — a **different** license and copyright holder
than the **DHB-License** (Bailey personal, `dhbailey@lbl.gov`) that governs
`dd_math.hpp` / `ff_math.hpp` (DDFUN derivatives). Accordingly:

- `third_party/include/qf_math.hpp` carries `SPDX-License-Identifier:
  LicenseRef-LBNL-BSD-License`, NOT `LicenseRef-DHB-License`.
- `LICENSES/LicenseRef-LBNL-BSD-License.txt` added (verbatim modified-BSD text
  from QD 2.3.24's `COPYING` + `BSD-LBNL-License.doc`).

**Deviation from the T3.0a task prompt (recorded per rule 8):** the prompt said
"Copy ff_math.hpp's license header verbatim, adjust names" (i.e. DHB-License).
That conflicts with the plan doc's own §T3.0a-kickoff instruction to "verify QF
port lineage (DDFUN vs QD — QF is modeled on QD's qd_real.cc, so likely
LBNL-BSD-License) and apply the correct license header." The plan-doc lineage
check wins: applying the DHB-License to a QD derivative would mis-attribute
copyright to Bailey personally and cite the wrong commercial contact. The
LBNL-BSD-License is applied instead. `NOTICE.md` should gain a QF row when QF
merges to `main` (a T3.x merge task, out of T3.0a scope).
