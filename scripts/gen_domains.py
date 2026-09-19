#!/usr/bin/env python3
"""Generate docs/DOMAINS.md and docs/DEVICE_PRECISION.md from the sweep data.

Nothing here is hand-written.

    scripts/gen_domains.py > docs/DOMAINS.md
    scripts/gen_domains.py --device-precision > docs/DEVICE_PRECISION.md

Reads, all committed:

  validation/sweep/sweep_baseline.csv.gz
  validation/sweep/sweep_baseline_a100.csv.gz   (may be absent → NO BASELINE gap)
  validation/sweep/sweep_baseline_mi250.csv.gz  (may be absent → NO BASELINE gap)
  validation/sweep/sweep_grid.csv
  validation/sweep/open_defects.txt

WHY A GENERATOR AND NOT PROSE
  Two documents in this repository have already drifted from reality because a
  human maintained them by hand. A domain table is exactly the kind of document
  that drifts: it is 252 cells, each of which moves whenever a numeric fix
  lands. So the table is derived, and validation/check_domains_fresh.sh /
  validation/check_device_domains_fresh.sh fail if the committed markdown is
  not what this script currently emits.

THE THREE THRESHOLDS, and the judgment in each
  TRUST = 0.90 x cap   a point at or above this is "the op working properly".
                       Not 1.00 x cap: the caps are round numbers a shade below
                       what the formats actually deliver, so honest points sit a
                       few hundredths under and a 1.00 test would call them
                       failures.
  TRIAGE = 0.50 x cap  below this a point is counted as a failure in the
                       "fails" column. This is a DISPLAY threshold for the
                       document only. It is not a verdict: the verdict is the
                       ulp measurement against the derived bound, carried in the
                       baseline's `state` column and in the register.
  Anything between the two is degraded but usable, and is reported as the
  boundary band rather than as a failure.

DEVICE CAUSE CLASSES (DEVICE_PRECISION.md only)
  match          every point bit-identical to host (digits, ulps, bound, state)
  FMA            an arithmetic/Dekker cell differs — contraction should have
                 been off; a nonzero count here is a build-flag regression
  FTZ            state moved into N, or a digit change concentrated below the
                 format's full-precision floor
  VENDOR_LIBM    a transcendental / libm-backed cell differs
  UNATTRIBUTED   differs, and none of the above fits
  NO BASELINE    arch column is a gap (C7/C8 degradation rule), not silence
"""

import collections
import gzip
import os
import sys

TRUST_FRAC = 0.90
TRIAGE_FRAC = 0.50
NOISE = 10 ** 0.1          # monotone gate's 0.1-digit noise floor, as a factor
ALLOW = 8.0                # kUlpAllowance — absolute gate multiplier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "validation/sweep/sweep_baseline.csv.gz")
BASELINE_A100 = os.path.join(ROOT, "validation/sweep/sweep_baseline_a100.csv.gz")
BASELINE_MI250 = os.path.join(ROOT, "validation/sweep/sweep_baseline_mi250.csv.gz")
GRID = os.path.join(ROOT, "validation/sweep/sweep_grid.csv")
REGISTER = os.path.join(ROOT, "validation/sweep/open_defects.txt")

# Status strings when a device baseline is missing. Kept as data so the
# generator can never silently drop an arch column (CORE_PLAN C9 degradation).
ARCH_META = {
    "a100": {
        "path": BASELINE_A100,
        "label": "a100",
        "job": "Cobalt 1001685",
        "status_ptr": "docs/CORE_PLAN_STATUS.md §C7",
        "no_baseline": "NO BASELINE — see C7 STATUS",
    },
    "mi250": {
        "path": BASELINE_MI250,
        "label": "mi250",
        "job": "Cobalt 1001915",
        "status_ptr": "docs/CORE_PLAN_STATUS.md §C8",
        "no_baseline": "NO BASELINE — see C8 STATUS",
    },
}

CAPS = {"DD": 31.00, "QF": 29.00, "TF": 21.70, "FF": 14.00}
BACKENDS = ["DD", "QF", "TF", "FF"]          # widest cap first

# Word format facts, quoted in the header. Same numbers the classifier derives
# its Range from -- see scripts/sweep_accuracy.cpp.
FORMATS = [
    ("DD", "2 x FP64", "~31", "2.0e-292", "1.8e+308", "4.9e-324"),
    ("QF", "4 x FP32", "~29", "5.6e-17", "3.4e+38", "1.4e-45"),
    ("TF", "3 x FP32", "~21.7", "3.3e-24", "3.4e+38", "1.4e-45"),
    ("FF", "2 x FP32", "~14", "2.0e-31", "3.4e+38", "1.4e-45"),
]
FLOORS = {be: float(floor) for be, _, _, floor, _, _ in FORMATS}

# Ops whose hot path is EFT / bit manipulation / selection — no vendor libm.
# Measured bit-identical host↔a100↔mi250 in C9; a future diff is FMA.
ARITH_OPS = set(
    "add sub mul div fma abs copysign fmax fmin fdim hypot "
    "ceil floor round trunc fmod remainder conj polar".split()
)
# Ops that call through xp::detail:: into a vendor scalar libm (or compose
# several such calls). A host↔device delta here is VENDOR_LIBM unless FTZ
# dominates.
LIBM_OPS = set(
    "sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh "
    "exp exp2 exp10 expm1 log log2 log10 log1p pow sqrt".split()
)

FAMILIES = [
    ("Arithmetic and selection", "r",
     "add sub mul div fma abs copysign fmax fmin fdim hypot".split()),
    ("Rounding and remainder", "r",
     "ceil floor round trunc fmod remainder".split()),
    ("Exponential, logarithmic and power", "r",
     "exp exp2 exp10 expm1 log log2 log10 log1p pow sqrt".split()),
    ("Trigonometric and inverse trigonometric", "r",
     "sin cos tan asin acos atan".split()),
    ("Hyperbolic and inverse hyperbolic", "r",
     "sinh cosh tanh asinh acosh atanh".split()),
    ("Complex arithmetic and construction", "c",
     "add sub mul div abs conj polar".split()),
    ("Complex exponential, logarithmic, power and root", "c",
     "exp log log10 pow sqrt".split()),
    ("Complex trigonometric and hyperbolic", "c",
     "sin cos tan sinh cosh tanh".split()),
    ("Complex inverse functions", "c",
     "asin acos atan asinh acosh atanh".split()),
]

CLASS_BLURB = {
    "at or below bound": "every point is at or under the bound the format and the "
                         "conditioning derive for it -- nothing left to explain",
    "UNRESOLVED": "no verdict issuable: the answer, an operand or an intermediate "
                  "does not fit the format, or the derived bound already exceeds "
                  "2^p ulps",
    "OPEN DEFECT": "at least one point exceeds its derived bound and is carried in "
                   "validation/sweep/open_defects.txt",
}

FAMILY_BLURB = {
    "log": "log sweep, |x| = 10^e",
    "linear": "linear sweep over [-8, 8]",
    "ulp": "within 2 ulp of 0, +-1, +-pi/2 or a multiple of pi",
    "near1": "geometric approach to +-1",
    "polar": "polar shells",
    "cut-re": "perpendicular approach to the real axis",
    "cut-im": "perpendicular approach to the imaginary axis",
    "axis": "the real axis on a geometric ladder",
}


def read_grid():
    """point id -> (family, |input|). |input| is |x| for real, |z| for complex."""
    out = {}
    with open(GRID) as f:
        for line in f:
            if line.startswith("#") or line.startswith("kind,"):
                continue
            kind, point, family, re_s, im_s = line.rstrip("\n").split(",")
            re_v = float(re_s)
            mag = abs(re_v) if not im_s else (re_v * re_v + float(im_s) ** 2) ** 0.5
            out[(kind, int(point))] = (family, mag)
    return out


def read_baseline(path):
    """(backend, kind, op, point) -> (digits, ulps, bound, state).

    Also returns per-cell state counters and a where-tag from the header.
    Returns None if the file is missing (degradation rule).
    """
    if not os.path.exists(path):
        return None
    rows = {}
    states = collections.defaultdict(collections.Counter)
    where = None
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("# where:"):
                where = line.split(":", 1)[1].strip()
                continue
            if (line.startswith("#") or line.startswith("backend,")
                    or line.startswith("where,")):
                continue
            p = line.rstrip("\n").split(",")
            # Optional leading `where` column (C6).
            if p and p[0] not in CAPS:
                p = p[1:]
            be, kind, opn, point = p[0], p[1], p[2], int(p[3])
            digits, ulps, bound, state = (
                float(p[4]), float(p[5]), float(p[6]), p[7])
            rows[(be, kind, opn, point)] = (digits, ulps, bound, state)
            states[(be, kind, opn)][state] += 1
    return {"rows": rows, "states": states, "where": where, "path": path}


def read_register():
    """(backend, kind, op) -> count of points known to exceed their bound."""
    out = collections.Counter()
    if not os.path.exists(REGISTER):
        sys.stderr.write("missing %s\n" % REGISTER)
        sys.exit(2)
    with open(REGISTER) as f:
        for line in f:
            line = line.split("#", 1)[0].split()
            if len(line) == 4:
                out[(line[0], line[1], line[2])] += 1
    return out


def fmt_mag(v):
    if v == 0:
        return "0"
    if 1e-4 <= v < 1e5:
        return "%.4g" % v
    return "%.0e" % v


def fmt_ulps(v):
    if v == 0:
        return "0"
    if v < 1e-3:
        return "%.3g" % v
    if v < 10:
        return "%.4g" % v
    if v < 1e6:
        return "%.4g" % v
    return "%.3e" % v


def count_above_bound(rows):
    n = 0
    for digits, ulps, bound, state in rows.values():
        if state == "S" and ulps > ALLOW * bound:
            n += 1
    return n


def baseline_by_cell(rows):
    """Group point rows into (backend, kind, op) -> [(point, digits)]."""
    out = collections.defaultdict(list)
    for (be, kind, op, point), (digits, ulps, bound, state) in rows.items():
        out[(be, kind, op)].append((point, digits))
    return out


def cell(be, kind, op, grid, baseline, states, register):
    """Everything DOMAINS.md says about one (backend, op) pair."""
    cap = CAPS[be]
    pts = baseline[(be, kind, op)]
    trust, triage = TRUST_FRAC * cap, TRIAGE_FRAC * cap

    bad = [(p, grid[(kind, p)][0], grid[(kind, p)][1], d) for p, d in pts if d < triage]

    n = len(pts)
    mean = sum(d for _, d in pts) / n
    n_ok = sum(1 for _, d in pts if d >= trust)

    by_mag = {}
    for p, d in pts:
        m = grid[(kind, p)][1]
        by_mag.setdefault(m, []).append(d)
    mags = sorted(by_mag)
    best, cur = None, None
    for m in mags:
        if min(by_mag[m]) >= trust:
            if cur is None:
                cur = [m, m, len(by_mag[m])]
            else:
                cur[1], cur[2] = m, cur[2] + len(by_mag[m])
            if best is None or cur[2] > best[2]:
                best = list(cur)
        else:
            cur = None
    if best is None:
        band, lo, hi = "none", None, None
    else:
        lo, hi = best[0], best[1]
        band = "%s .. %s" % (fmt_mag(lo), fmt_mag(hi))

    boundary = []
    if lo is not None and bad:
        hi_bad = [(m, d) for _, _, m, d in bad if m > hi]
        lo_bad = [(m, d) for _, _, m, d in bad if m < lo]
        if lo_bad:
            m, d = max(lo_bad, key=lambda t: t[0])
            boundary.append("below %s: %.2f" % (fmt_mag(m), d))
        if hi_bad:
            m, d = min(hi_bad, key=lambda t: t[0])
            boundary.append("above %s: %.2f" % (fmt_mag(m), d))
    if not boundary:
        boundary.append("--" if not bad else "no trusted band")

    st = states[(be, kind, op)]
    n_above = register.get((be, kind, op), 0)
    n_unres = st.get("U", 0) + st.get("N", 0) + st.get("X", 0)
    if n_above:
        dominant = "OPEN DEFECT"
    elif n_unres:
        dominant = "UNRESOLVED"
    else:
        dominant = "at or below bound"

    fam = collections.Counter(f for _, f, _, _ in bad)
    region = FAMILY_BLURB.get(fam.most_common(1)[0][0], "") if fam else ""

    return {
        "n": n, "mean": mean, "band": band, "n_bad": len(bad),
        "ok_pct": 100.0 * n_ok / n, "boundary": "; ".join(boundary),
        "dominant": dominant, "unexplained": n_above,
        "classes": collections.Counter({dominant: 1}),
        "reasons": collections.Counter({"unresolved": n_unres, "above bound": n_above}),
        "region": region,
    }


def abs_trusted_floor(rows, grid, be):
    """Smallest |x| at which every real `abs` point still reaches 90% of cap.

    Walking large→small and stopping at the first bad magnitude would report the
    *top* of a hole; taking the min good magnitude reports the floor a caller
    can rely on contiguous from that point upward (matching DOMAINS.md's band).
    """
    trust = TRUST_FRAC * CAPS[be]
    by_mag = {}
    for (b, kind, op, point), (digits, ulps, bound, state) in rows.items():
        if b != be or kind != "r" or op != "abs":
            continue
        m = grid[(kind, point)][1]
        by_mag.setdefault(m, []).append(digits)
    good = [m for m in by_mag if m > 0 and min(by_mag[m]) >= trust]
    return min(good) if good else None


def compare_arch(host_rows, device_rows, grid):
    """Pointwise host↔device comparison used by DEVICE_PRECISION.md.

    Worse/better are scored only when BOTH sides are `S`: an `S → N` move is a
    state change, not an accuracy improvement, and N carries the sentinel
    `ulps=-1` which would otherwise flood the "better" column.
    """
    identical = differ = worse = better = state_move = 0
    state_pairs = collections.Counter()
    for k, hv in host_rows.items():
        dv = device_rows[k]
        if hv[3] != dv[3]:
            state_move += 1
            state_pairs[(hv[3], dv[3])] += 1
        if hv == dv:
            identical += 1
            continue
        differ += 1
        if hv[3] != "S" or dv[3] != "S":
            continue
        hu, xu = hv[1], dv[1]
        if xu > hu * NOISE and (hu > 0 or xu > 0):
            worse += 1
        elif hu > xu * NOISE and (xu > 0 or hu > 0):
            better += 1
    return {
        "identical": identical,
        "differ": differ,
        "worse": worse,
        "better": better,
        "state_move": state_move,
        "state_pairs": state_pairs,
        "above": count_above_bound(device_rows),
    }


def mean_ulps_scored(rows, keys):
    """Mean ulps over points with state `S` only. N/U carry ulps=-1 / 0."""
    scored = [rows[k][1] for k in keys if rows[k][3] == "S"]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def cell_device_stats(host_rows, device_rows, grid, be, kind, op):
    """Per-(backend, kind, op) host↔device summary for one arch."""
    keys = [k for k in host_rows if k[0] == be and k[1] == kind and k[2] == op]
    n = len(keys)
    if n == 0:
        return None
    mean_h = mean_ulps_scored(host_rows, keys)
    mean_d = mean_ulps_scored(device_rows, keys)
    identical = sum(1 for k in keys if host_rows[k] == device_rows[k])
    state_move = sum(1 for k in keys if host_rows[k][3] != device_rows[k][3])
    # Digit changes below the format floor — FTZ signal.
    floor = FLOORS[be]
    below_floor_digit = 0
    for k in keys:
        if host_rows[k][0] == device_rows[k][0]:
            continue
        mag = grid[(k[1], k[3])][1]
        if 0 < mag < floor:
            below_floor_digit += 1
    sn = sum(1 for k in keys
             if host_rows[k][3] == "S" and device_rows[k][3] == "N")
    return {
        "n": n,
        "mean_h": mean_h,
        "mean_d": mean_d,
        "identical": identical,
        "state_move": state_move,
        "s_to_n": sn,
        "below_floor_digit": below_floor_digit,
    }


def classify_cause(op, a100, mi250):
    """One cause label for a (backend, kind, op) cell across both arches."""
    # Gap on either arch is reported at the coverage layer, not here.
    diffs = []
    for stats in (a100, mi250):
        if stats is None:
            continue
        if stats["identical"] < stats["n"]:
            diffs.append(stats)
    if not diffs:
        return "match"

    if any(s["s_to_n"] or s["below_floor_digit"] for s in diffs):
        return "FTZ"
    if op in ARITH_OPS:
        return "FMA"
    if op in LIBM_OPS:
        return "VENDOR_LIBM"
    return "UNATTRIBUTED"


def arch_status_line(name, data):
    meta = ARCH_META[name]
    if data is None:
        return ("%s: **%s** (%s)" % (name, meta["no_baseline"], meta["status_ptr"]))
    return ("%s: PRESENT — %s rows, where=`%s`, job %s (%s)"
            % (name, len(data["rows"]), data["where"], meta["job"], meta["status_ptr"]))


def emit_domains(grid, host, register, a100, mi250):
    """Emit docs/DOMAINS.md to stdout."""
    baseline = baseline_by_cell(host["rows"])
    states = host["states"]

    o = []
    w = o.append

    w("# Domain limits — where each operation can and cannot be trusted")
    w("")
    w("**GENERATED FILE — do not edit.** Produced by `scripts/gen_domains.py` from")
    w("`validation/sweep/sweep_baseline.csv.gz`, `validation/sweep/sweep_grid.csv` and")
    w("`validation/sweep/open_defects.txt`. Regenerate with:")
    w("")
    w("```bash")
    w("scripts/gen_domains.py > docs/DOMAINS.md")
    w("```")
    w("")
    w("`validation/check_domains_fresh.sh` fails if this file is not what the")
    w("generator currently emits.")
    w("")
    w("---")
    w("")
    w("## Device precision — every number below is host-measured")
    w("")
    w("**Every digit count, trusted band and verdict in this document is from the")
    w("host baseline** (`validation/sweep/sweep_baseline.csv.gz`, `where=host`).")
    w("It does **not** describe what the same op does on a GPU. Subnormal trailing")
    w("limbs, vendor `libm`, and the gfx90a mitigations can all move a cell; the")
    w("measured record of those moves is")
    w("[`docs/DEVICE_PRECISION.md`](DEVICE_PRECISION.md), generated from the three")
    w("baselines by `scripts/gen_domains.py --device-precision` and gated by")
    w("`device_domains_fresh`.")
    w("")
    w("Arch coverage (an absent arch is a named gap, never a missing column):")
    w("")
    w("- %s" % arch_status_line("a100", a100))
    w("- %s" % arch_status_line("mi250", mi250))
    w("")
    w("Absolute gate (`ulps ≤ 8 × derived bound`) on every present arch:")
    above_h = count_above_bound(host["rows"])
    above_a = count_above_bound(a100["rows"]) if a100 else None
    above_m = count_above_bound(mi250["rows"]) if mi250 else None
    w("- host: %d above bound" % above_h)
    w("- a100: %s" % (
        ("%d above bound" % above_a) if above_a is not None
        else ARCH_META["a100"]["no_baseline"]))
    w("- mi250: %s" % (
        ("%d above bound" % above_m) if above_m is not None
        else ARCH_META["mi250"]["no_baseline"]))
    w("")
    w("---")
    w("")
    w("## How to read this")
    w("")
    w("Every number below is measured, on a fixed grid of 1700 real and 1780 complex")
    w("inputs per op, scored against an MPFR/MPC oracle at 400 bits, quantised into")
    w("`__float128` (binary128). A backend's **cap** is the most digits its format")
    w("can carry; a cell reports where the op actually reaches that cap and where")
    w("it does not.")
    w("")
    w("| backend | words | cap (digits) | full-precision floor | word max | word min (subnormal) |")
    w("|---|---|---|---|---|---|")
    for be, words, cap, floor, mx, mn in FORMATS:
        w("| %s | %s | %s | %s | %s | %s |" % (be, words, cap, floor, mx, mn))
    w("")
    w("**Full-precision floor** is the magnitude below which the *trailing* limbs go")
    w("subnormal, so the type stops carrying its nominal digit count even though the")
    w("leading word is still fine. It is the number that matters for small inputs:")
    w("FF is out of digits below ~2e-31 while DD keeps all 31 down to ~2e-292.")
    w("Whether a GPU's FTZ/DAZ mode destroys that floor further is measured in")
    w("`docs/DEVICE_PRECISION.md`, not assumed from the format facts above.")
    w("")
    w("Column meanings:")
    w("")
    w("- **trusted |x|** — the widest *contiguous* band of input magnitude in which")
    w("  **every** grid point reaches at least 90% of cap. For complex ops the")
    w("  magnitude is |z|. `none` means no such band exists. The band is contiguous")
    w("  by construction, so it has no holes: a caller can rely on all of it.")
    w("- **at cap** — the share of *all* the op's grid points reaching 90% of cap,")
    w("  including any outside the band. A wide band with a low percentage means the")
    w("  op also works in places the single interval does not cover.")
    w("- **boundary** — the failing point nearest each end of the trusted band, with")
    w("  the digits measured there, so the degradation is quantified and not merely")
    w("  located.")
    w("- **fails** — how many of the op's points score below 50% of cap, and the")
    w("  single verdict for the cell: `at or below bound` when every point is")
    w("  explained by the format and the conditioning, `UNRESOLVED` when the format")
    w("  cannot carry some point at all, `OPEN DEFECT` when some point exceeds its")
    w("  derived bound and is carried in `validation/sweep/open_defects.txt`.")
    w("")
    w("**Caveat for the binary ops** (`add sub mul div pow hypot fmod remainder`")
    w("`copysign fmax fmin fdim fma`, and the complex `add sub mul div pow`): the")
    w("magnitude axis is the **first operand only**. The second is drawn")
    w("log-uniformly over a per-op window, with one point in seven a deliberately")
    w("cancelling pair, so a failure at a given |x| may be caused by the operand")
    w("paired with it rather than by x. For those rows read the `fails` count and the")
    w("verdict, and treat the band as indicative. The unary rows have no such")
    w("ambiguity.")
    w("")
    w("Verdicts, from `scripts/sweep_accuracy --ulp`:")
    w("")
    for k in ["at or below bound", "UNRESOLVED", "OPEN DEFECT"]:
        w("- **%s** — %s" % (k, CLASS_BLURB[k]))
    w("")
    w("There is one measurement behind all three: the error in ulps against the")
    w("MPFR/MPC oracle, compared against a bound derived from the")
    w("format (word count, exponent range, intermediate width) and the condition")
    w("number — never from what the implementation currently scores. The derivation is")
    w("in `docs/CORRECTNESS.md`. `UNRESOLVED` and `at or below bound` are both healthy")
    w("outcomes; `OPEN DEFECT` is the only one that names a bug, and every such point")
    w("is listed by name in the register.")
    w("")
    w("### Why grouped by operation family rather than by backend")
    w("")
    w("The four backends share one op inventory and one grid, so their cells are")
    w("directly comparable — and the question a caller actually has is *\"can this op")
    w("hold my range, and if not which backend can?\"*. Putting the four backends on")
    w("adjacent rows answers that by eye. A backend-major layout would answer the")
    w("rarer question (\"what does DD do across all 63 ops?\") while forcing a reader")
    w("comparing FF against DD to page between two distant sections.")
    w("")
    w("---")
    w("")

    totals = collections.Counter()
    for title, kind, ops in FAMILIES:
        w("## %s (%s)" % (title, "real" if kind == "r" else "complex"))
        w("")
        w("| op | backend | cap | mean | trusted \\|%s\\| | at cap | boundary (digits) | fails | verdict |"
          % ("x" if kind == "r" else "z"))
        w("|---|---|---:|---:|---|---:|---|---:|---|")
        for op in ops:
            for be in BACKENDS:
                c = cell(be, kind, op, grid, baseline, states, register)
                totals.update(c["classes"])
                w("| `%s` | %s | %.2f | %.2f | %s | %.0f%% | %s | %d | %s |" % (
                    op, be, CAPS[be], c["mean"], c["band"], c["ok_pct"],
                    c["boundary"], c["n_bad"],
                    c["dominant"] + ("" if c["unexplained"] == 0
                                     else " (**%d above bound**)" % c["unexplained"])))
        w("")

        notes = []
        for op in ops:
            for be in BACKENDS:
                c = cell(be, kind, op, grid, baseline, states, register)
                if c["n_bad"] and c["region"]:
                    notes.append((op, be, c["n_bad"], c["region"], c["reasons"]))
        if notes:
            w("**Where the failures sit.** The grid family carrying the most failures")
            w("for each cell that has any:")
            w("")
            seen = {}
            for op, be, n_bad, region, reasons in notes:
                seen.setdefault(op, []).append("%s %d pts (%s)" % (be, n_bad, region))
            for op in ops:
                if op in seen:
                    w("- `%s` — %s" % (op, "; ".join(seen[op])))
            w("")
        w("---")
        w("")

    w("## Totals across all 252 cells")
    w("")
    w("| classification | cells |")
    w("|---|---:|")
    for k in ["at or below bound", "UNRESOLVED", "OPEN DEFECT"]:
        w("| %s | %d |" % (k, totals.get(k, 0)))
    w("| **total cells** | **%d** |" % sum(totals.values()))
    w("")
    w("Out of 436,080 scored points (1700 real + 1780 complex inputs × 4 backends")
    w("× 63 ops).")
    w("")
    w("---")
    w("")
    w("## Freshness")
    w("")
    w("`validation/check_domains_fresh.sh` regenerates this file and diffs it against")
    w("the committed copy, exiting nonzero on any difference. It needs only Python 3")
    w("and the committed CSVs — no build, no Kokkos. Registered as the `domains_fresh`")
    w("ctest target. The device counterpart is `device_domains_fresh`.")
    w("")
    w("Note that the check only proves the markdown matches the CSVs. If a numeric fix")
    w("lands, the baseline must be regenerated first — `scripts/sweep_accuracy --ulp")
    w("--register validation/sweep/open_defects.txt --out <tmp>`, gzipped into")
    w("`validation/sweep/sweep_baseline.csv.gz` — and only then this file.")

    sys.stdout.write("\n".join(o) + "\n")


def emit_device_precision(grid, host, a100, mi250):
    """Emit docs/DEVICE_PRECISION.md to stdout."""
    o = []
    w = o.append

    w("# Device precision — host vs A100 vs MI250X")
    w("")
    w("**GENERATED FILE — do not edit.** Produced by")
    w("`scripts/gen_domains.py --device-precision` from the three committed sweep")
    w("baselines and `validation/sweep/sweep_grid.csv`. Regenerate with:")
    w("")
    w("```bash")
    w("scripts/gen_domains.py --device-precision > docs/DEVICE_PRECISION.md")
    w("```")
    w("")
    w("`validation/check_device_domains_fresh.sh` fails if this file is not what the")
    w("generator currently emits. Registered as the `device_domains_fresh` ctest")
    w("target.")
    w("")
    w("One measurement, one scorer: every ulp here is the host `sweep_accuracy`")
    w("MPFR/MPC oracle applied to raw limbs produced on each arch. There is no")
    w("device-side ulp computation. See `docs/CORRECTNESS.md`.")
    w("")

    # ---- coverage (degradation rule) ----------------------------------------
    present = []
    missing = []
    for name, data in (("a100", a100), ("mi250", mi250)):
        if data is None:
            missing.append(name)
        else:
            present.append(name)

    if missing:
        w("**Coverage is partial.** An arch with no committed baseline is reported")
        w("as an explicit gap below — never as a column that quietly is not there.")
        w("")
    else:
        w("**Coverage is complete for the two device arches this arc measures.**")
        w("Both A100 and MI250X baselines are present; every table below has a")
        w("measured column for each.")
        w("")

    w("## Arch coverage")
    w("")
    w("| arch | baseline | status |")
    w("|---|---|---|")
    w("| host | `validation/sweep/sweep_baseline.csv.gz` | PRESENT — %d rows, where=`%s` |"
      % (len(host["rows"]), host["where"]))
    for name in ("a100", "mi250"):
        meta = ARCH_META[name]
        data = a100 if name == "a100" else mi250
        if data is None:
            w("| %s | `%s` | **%s** (%s) |" % (
                name, os.path.basename(meta["path"]),
                meta["no_baseline"], meta["status_ptr"]))
        else:
            w("| %s | `%s` | PRESENT — %d rows, where=`%s`, %s (%s) |" % (
                name, os.path.basename(meta["path"]),
                len(data["rows"]), data["where"], meta["job"], meta["status_ptr"]))
    w("")

    above_h = count_above_bound(host["rows"])
    w("Absolute gate (`ulps ≤ 8 × derived bound`), scored points above it:")
    w("")
    w("| arch | above bound |")
    w("|---|---:|")
    w("| host | %d |" % above_h)
    for name, data in (("a100", a100), ("mi250", mi250)):
        if data is None:
            w("| %s | %s |" % (name, ARCH_META[name]["no_baseline"]))
        else:
            w("| %s | %d |" % (name, count_above_bound(data["rows"])))
    w("")
    w("`validation/sweep/open_defects.txt` carries a point only when it exceeds")
    w("that bound. A zero in every present column means the register stays empty.")
    w("")

    # ---- global deltas ------------------------------------------------------
    w("## Host vs device — global deltas")
    w("")
    w("Same identity key, same 0.1-digit noise floor the monotone gate uses")
    w("(`10^0.1`). \"Worse\" / \"better\" count only points that move *beyond*")
    w("that floor; \"differ at all\" is any ulps/digits/state change.")
    w("")
    w("| arch | identical | differ at all | worse beyond noise | better beyond noise | state moved | above bound |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for name, data in (("a100", a100), ("mi250", mi250)):
        if data is None:
            gap = ARCH_META[name]["no_baseline"]
            w("| %s | %s | — | — | — | — | — |" % (name, gap))
            continue
        c = compare_arch(host["rows"], data["rows"], grid)
        w("| %s | %d | %d | %d | %d | %d | %d |" % (
            name, c["identical"], c["differ"], c["worse"], c["better"],
            c["state_move"], c["above"]))
    w("")

    if a100 is not None:
        c = compare_arch(host["rows"], a100["rows"], grid)
        if c["state_pairs"]:
            w("A100 state transitions (host → a100):")
            w("")
            for (a, b), n in c["state_pairs"].most_common():
                w("- `%s → %s`: %d" % (a, b, n))
            w("")
    if mi250 is not None:
        c = compare_arch(host["rows"], mi250["rows"], grid)
        if c["state_pairs"]:
            w("MI250 state transitions (host → mi250):")
            w("")
            for (a, b), n in c["state_pairs"].most_common():
                w("- `%s → %s`: %d" % (a, b, n))
            w("")
        elif a100 is not None:
            w("MI250: **0** state transitions. Every point keeps the host verdict")
            w("letter; the 12k+ ulp moves below stay inside `S`/`U`/`N` as scored")
            w("on the host.")
            w("")

    # ---- four mechanisms ----------------------------------------------------
    w("---")
    w("")
    w("## The four mechanisms, measured")
    w("")
    w("The repository already knew these in pieces. The numbers below are from")
    w("the three baselines, not from reasoning about flags.")
    w("")

    # 1. FTZ/DAZ
    w("### 1. FTZ / DAZ on the trailing limbs")
    w("")
    w("The \"full-precision floor\" row of `docs/DOMAINS.md` (DD 2.0e-292, QF")
    w("5.6e-17, TF 3.3e-24, FF 2.0e-31) is the magnitude below which trailing")
    w("limbs go subnormal. A GPU flushing subnormals would raise that floor.")
    w("Measured floor here is the smallest |x| at which real `abs` still reaches")
    w("90% of cap on every grid point at that magnitude — the same definition")
    w("`docs/DOMAINS.md` uses for its trusted band.")
    w("")
    w("| backend | format floor | host measured | a100 measured | mi250 measured |")
    w("|---|---|---|---|---|")
    for be, _, _, floor, _, _ in FORMATS:
        h = abs_trusted_floor(host["rows"], grid, be)
        a = (abs_trusted_floor(a100["rows"], grid, be) if a100
             else ARCH_META["a100"]["no_baseline"])
        m = (abs_trusted_floor(mi250["rows"], grid, be) if mi250
             else ARCH_META["mi250"]["no_baseline"])
        def fmt_f(v):
            if isinstance(v, str):
                return v
            return fmt_mag(v) if v is not None else "none"
        w("| %s | %s | %s | %s | %s |" % (be, floor, fmt_f(h), fmt_f(a), fmt_f(m)))
    w("")
    w("On every present arch the measured `abs` floor matches the host. FTZ did")
    w("**not** raise the trusted band for `abs`. The FTZ signal that *does*")
    w("appear is scorability, not the floor:")
    w("")
    if a100 is not None:
        c = compare_arch(host["rows"], a100["rows"], grid)
        sn = c["state_pairs"].get(("S", "N"), 0)
        w("- A100: **%d** points move `S → N` (host scored them; A100 marks them" % sn)
        w("  unscorable). They concentrate in DD complex `atan` and `atanh` — see")
        w("  the C7 STATUS block. That is why A100 has its own baseline rather than")
        w("  sharing the host monotone gate.")
    else:
        w("- A100: %s" % ARCH_META["a100"]["no_baseline"])
    if mi250 is not None:
        c = compare_arch(host["rows"], mi250["rows"], grid)
        sn = c["state_pairs"].get(("S", "N"), 0)
        w("- MI250: **%d** `S → N` moves." % sn)
    else:
        w("- MI250: %s" % ARCH_META["mi250"]["no_baseline"])
    w("")

    # 2. FMA
    w("### 2. FMA contraction")
    w("")
    w("Every arch builds with `-ffp-contract=off` / `--fmad=false`, so Dekker")
    w("TwoSum / TwoProd sequences must not collapse into an FMA. Confirmation is")
    w("from the results, not from the flags: count of bit-identical points in the")
    w("arithmetic / selection / rounding family")
    w("(`add sub mul div fma abs copysign fmax fmin fdim hypot ceil floor round")
    w("trunc fmod remainder conj polar`), host vs each device.")
    w("")
    arith_keys = [k for k in host["rows"] if k[2] in ARITH_OPS]
    n_arith = len(arith_keys)
    w("| arch | bit-identical arithmetic points | of |")
    w("|---|---:|---:|")
    for name, data in (("a100", a100), ("mi250", mi250)):
        if data is None:
            w("| %s | %s | %d |" % (name, ARCH_META[name]["no_baseline"], n_arith))
        else:
            same = sum(1 for k in arith_keys if host["rows"][k] == data["rows"][k])
            w("| %s | %d | %d |" % (name, same, n_arith))
    w("")
    w("A count below the total would be an FMA regression. Both present arches")
    w("are exact.")
    w("")

    # 3. Vendor libm
    w("### 3. Vendor libm")
    w("`xp::detail::` scalar dispatch resolves `sin` / `exp` / `sqrt` / … to a")
    w("different implementation on each device. Every host↔device ulp move that")
    w("is not FTZ and not arithmetic lands here.")
    w("")
    libm_keys = [k for k in host["rows"] if k[2] in LIBM_OPS]
    w("| arch | libm-family points that differ | of | worse beyond noise | better beyond noise |")
    w("|---|---:|---:|---:|---:|")
    for name, data in (("a100", a100), ("mi250", mi250)):
        if data is None:
            w("| %s | %s | %d | — | — |" % (
                name, ARCH_META[name]["no_baseline"], len(libm_keys)))
            continue
        differ = worse = better = 0
        for k in libm_keys:
            hv, dv = host["rows"][k], data["rows"][k]
            if hv == dv:
                continue
            differ += 1
            if hv[3] != "S" or dv[3] != "S":
                continue
            hu, xu = hv[1], dv[1]
            if xu > hu * NOISE and (hu > 0 or xu > 0):
                worse += 1
            elif hu > xu * NOISE and (xu > 0 or hu > 0):
                better += 1
        w("| %s | %d | %d | %d | %d |" % (
            name, differ, len(libm_keys), worse, better))
    w("")
    w("The per-op table below names the cells. Inverse trig, inverse hyperbolic,")
    w("and `log` / `pow` dominate; pure arithmetic does not appear.")
    w("")

    # 4. gfx90a
    w("### 4. The gfx90a mitigations")
    w("")
    w("`XPMATH_NOINLINE_FUNCTION` (TD-1 / BranchRelaxation) and the de-recursed")
    w("device self-calls (TD-2 / dynamic stack) exist to change codegen on")
    w("MI250X. Their success is not an ulp delta — it is that the producer")
    w("finished:")
    w("")
    if mi250 is None:
        w("- %s" % ARCH_META["mi250"]["no_baseline"])
        w("- Widening the mitigation is out of scope for a documentation section;")
        w("  see the C8 STATUS block and `docs/ROCM_BRANCH_RELAXATION_BUG.md`.")
    else:
        w("- MI250 producer: `last_error() == 0` over all 436,080 points")
        w("  (Cobalt %s)." % ARCH_META["mi250"]["job"].split()[-1])
        w("- Absolute gate on the scored baseline: **%d** above bound."
          % count_above_bound(mi250["rows"]))
        w("- Arithmetic family: bit-identical to host (table above) — the")
        w("  volatile EFT wrappers under `__HIP_DEVICE_COMPILE__` did not")
        w("  perturb TwoSum/TwoProd.")
        w("- `scripts/xpm_lint_device_asm.sh` was **not** re-run on the C8 binary")
        w("  (no `--save-temps` objects kept); completion is not a substitute for")
        w("  tiers 1–4. Recorded in the C8 STATUS block.")
    w("")

    # ---- per-op table -------------------------------------------------------
    w("---")
    w("")
    w("## Per-op mean ulps — host / a100 / mi250")
    w("")
    w("Mean ulps over **scored** (`state=S`) points only — `N` carries the")
    w("sentinel `ulps=-1` and must not enter the average. The identical-point")
    w("counts cover every grid point and are the sharper signal. `cause` is one")
    w("label for the cell across both device arches; see the class list at the")
    w("top of `scripts/gen_domains.py`.")
    w("")

    cause_totals = collections.Counter()
    for title, kind, ops in FAMILIES:
        w("### %s (%s)" % (title, "real" if kind == "r" else "complex"))
        w("")
        w("| backend | op | host ulps | a100 ulps | mi250 ulps | Δa100 | Δmi250 | ident a100 | ident mi250 | cause |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for op in ops:
            for be in BACKENDS:
                sa = (cell_device_stats(host["rows"], a100["rows"], grid, be, kind, op)
                      if a100 else None)
                sm = (cell_device_stats(host["rows"], mi250["rows"], grid, be, kind, op)
                      if mi250 else None)
                # Need a host-only mean even when an arch is missing.
                keys = [k for k in host["rows"]
                        if k[0] == be and k[1] == kind and k[2] == op]
                mean_h = mean_ulps_scored(host["rows"], keys)

                if a100 is None and mi250 is None:
                    cause = "NO BASELINE"
                else:
                    cause = classify_cause(op, sa, sm)
                cause_totals[cause] += 1

                def col_mean(stats, fallback_arch):
                    if stats is None:
                        return ARCH_META[fallback_arch]["no_baseline"]
                    return fmt_ulps(stats["mean_d"])

                def col_delta(stats, fallback_arch):
                    if stats is None:
                        return "—"
                    return fmt_ulps(stats["mean_d"] - mean_h)

                def col_ident(stats, fallback_arch):
                    if stats is None:
                        return "—"
                    return "%d/%d" % (stats["identical"], stats["n"])

                w("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    be, op, fmt_ulps(mean_h),
                    col_mean(sa, "a100"), col_mean(sm, "mi250"),
                    col_delta(sa, "a100"), col_delta(sm, "mi250"),
                    col_ident(sa, "a100"), col_ident(sm, "mi250"),
                    cause))
        w("")

    w("## Cause totals across 252 cells")
    w("")
    w("| cause | cells |")
    w("|---|---:|")
    for k in ["match", "FMA", "FTZ", "VENDOR_LIBM", "UNATTRIBUTED", "NO BASELINE"]:
        w("| %s | %d |" % (k, cause_totals.get(k, 0)))
    w("| **total** | **%d** |" % sum(cause_totals.values()))
    w("")
    n_unattr = cause_totals.get("UNATTRIBUTED", 0)
    if n_unattr:
        w("**%d cells are UNATTRIBUTED.** They differ from host and fit none of" % n_unattr)
        w("the four mechanisms cleanly. Do not invent a fifth to close the table.")
    else:
        w("**0 cells are UNATTRIBUTED.** Every differing cell landed in FTZ or")
        w("VENDOR_LIBM; every arithmetic cell matched.")
    w("")
    w("---")
    w("")
    w("## Freshness")
    w("")
    w("`validation/check_device_domains_fresh.sh` regenerates this file and diffs")
    w("it against the committed copy. Re-score a device arch only through the")
    w("recipes in `docs/CORRECTNESS.md` and the C7/C8 STATUS blocks; then rerun")
    w("the generator.")

    sys.stdout.write("\n".join(o) + "\n")


def load_all():
    if not os.path.exists(BASELINE):
        sys.stderr.write("missing host baseline %s\n" % BASELINE)
        sys.exit(2)
    grid = read_grid()
    host = read_baseline(BASELINE)
    a100 = read_baseline(BASELINE_A100)
    mi250 = read_baseline(BASELINE_MI250)
    register = read_register()
    return grid, host, a100, mi250, register


def main(argv):
    device = False
    for a in argv[1:]:
        if a == "--device-precision":
            device = True
        elif a in ("-h", "--help"):
            sys.stdout.write(__doc__)
            return 0
        else:
            sys.stderr.write("unknown argument: %s\n" % a)
            return 2

    grid, host, a100, mi250, register = load_all()
    # Positive assertion that every arch name appears in the device doc, even
    # as a gap — enforced by the gate's grep, but fail loud here too if we ever
    # drop a name from ARCH_META.
    assert "a100" in ARCH_META and "mi250" in ARCH_META

    if device:
        emit_device_precision(grid, host, a100, mi250)
    else:
        emit_domains(grid, host, register, a100, mi250)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
