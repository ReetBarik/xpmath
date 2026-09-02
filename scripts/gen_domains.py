#!/usr/bin/env python3
"""Generate docs/DOMAINS.md from the sweep data. Nothing here is hand-written.

    scripts/gen_domains.py > docs/DOMAINS.md

Reads three files, all committed:

  validation/sweep/sweep_baseline.csv    digits scored at every (backend, op, point)
  validation/sweep/sweep_grid.csv        the input each `point` id refers to
  validation/sweep/sweep_classified.csv  why each low-scoring point fails
                                         (scripts/sweep_accuracy --classify)

WHY A GENERATOR AND NOT PROSE
  Two documents in this repository have already drifted from reality because a
  human maintained them by hand. A domain table is exactly the kind of document
  that drifts: it is 252 cells, each of which moves whenever a numeric fix
  lands. So the table is derived, and validation/check_domains_fresh.sh fails if
  the committed markdown is not what this script currently emits.

THE THREE THRESHOLDS, and the judgment in each
  TRUST = 0.90 x cap   a point at or above this is "the op working properly".
                       Not 1.00 x cap: the caps are round numbers a shade below
                       what the formats actually deliver, so honest points sit a
                       few hundredths under and a 1.00 test would call them
                       failures.
  TRIAGE = 0.50 x cap  below this a point is triaged and classified. Same
                       threshold the classifier uses; changing it here alone
                       would make the "below 50%" counts disagree with
                       sweep_classified.csv.
  Anything between the two is degraded but usable, and is reported as the
  boundary band rather than as a failure.
"""

import collections
import os
import sys

TRUST_FRAC = 0.90
TRIAGE_FRAC = 0.50

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "validation/sweep/sweep_baseline.csv")
GRID = os.path.join(ROOT, "validation/sweep/sweep_grid.csv")
CLASSED = os.path.join(ROOT, "validation/sweep/sweep_classified.csv")

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
    "UNDERFLOW": "true result below the smallest magnitude the words hold",
    "OVERFLOW": "true result above the largest magnitude the words hold",
    "ARG_RANGE": "the input itself is outside what the words hold",
    "CONDITIONING": "measured ill-conditioning: no implementation at this precision does better",
    "UNEXPLAINED": "NOT explained by range or conditioning -- see docs/KNOWN_ISSUES.md",
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


def read_baseline():
    rows = collections.defaultdict(list)      # (backend, kind, op) -> [(point, digits)]
    with open(BASELINE) as f:
        for line in f:
            if line.startswith("#") or line.startswith("backend,"):
                continue
            be, kind, op, point, digits = line.rstrip("\n").split(",")
            rows[(be, kind, op)].append((int(point), float(digits)))
    return rows


def read_classified():
    """(backend, kind, op) -> {class: count}, and -> {reason: count}."""
    by_class = collections.defaultdict(collections.Counter)
    by_reason = collections.defaultdict(collections.Counter)
    if not os.path.exists(CLASSED):
        sys.stderr.write("missing %s -- run scripts/sweep_accuracy --classify\n" % CLASSED)
        sys.exit(2)
    with open(CLASSED) as f:
        for line in f:
            if line.startswith("#") or line.startswith("backend,"):
                continue
            p = line.rstrip("\n").split(",")
            key = (p[0], p[1], p[2])
            by_class[key][p[7]] += 1
            by_reason[key][p[8]] += 1
    return by_class, by_reason


def fmt_mag(v):
    if v == 0:
        return "0"
    if 1e-4 <= v < 1e5:
        s = ("%.4g" % v)
        return s
    return "%.0e" % v


def cell(be, kind, op, grid, baseline, by_class, by_reason):
    """Everything DOMAINS.md says about one (backend, op) pair."""
    cap = CAPS[be]
    pts = baseline[(be, kind, op)]
    trust, triage = TRUST_FRAC * cap, TRIAGE_FRAC * cap

    bad = [(p, grid[(kind, p)][0], grid[(kind, p)][1], d) for p, d in pts if d < triage]

    n = len(pts)
    mean = sum(d for _, d in pts) / n
    n_ok = sum(1 for _, d in pts if d >= trust)

    # Trusted band: the widest CONTIGUOUS interval of input magnitude in which
    # every grid point reaches 90% of cap.
    #
    # Taking min..max over the good points instead would be worthless: a single
    # good point out at 1e30 (exp(-1e30) correctly returns 0, and scores cap for
    # it) would stretch the band across the whole grid while everything from 300
    # upward is broken. So magnitudes are grouped, a magnitude counts as good
    # only if EVERY point at it is good, and the reported band is the run of
    # consecutive good magnitudes holding the most points. A caller can rely on
    # the whole of that interval; that is the property the column claims.
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

    # Degradation boundary: the failing point closest in magnitude to the top of
    # the trusted band, i.e. where it stops working as |x| grows, and the same at
    # the bottom. Reported with the digits actually measured there.
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

    classes = by_class[(be, kind, op)]
    dominant = classes.most_common(1)[0][0] if classes else "--"
    unexplained = classes.get("UNEXPLAINED", 0)

    # Where the failures sit, by grid family -- this is the "excluded region".
    fam = collections.Counter(f for _, f, _, _ in bad)
    region = FAMILY_BLURB.get(fam.most_common(1)[0][0], "") if fam else ""

    return {
        "n": n, "mean": mean, "band": band, "n_bad": len(bad),
        "ok_pct": 100.0 * n_ok / n, "boundary": "; ".join(boundary),
        "dominant": dominant, "unexplained": unexplained,
        "classes": classes, "reasons": by_reason[(be, kind, op)], "region": region,
    }


def main():
    grid = read_grid()
    baseline = read_baseline()
    by_class, by_reason = read_classified()

    o = []
    w = o.append

    w("# Domain limits — where each operation can and cannot be trusted")
    w("")
    w("**GENERATED FILE — do not edit.** Produced by `scripts/gen_domains.py` from")
    w("`validation/sweep/sweep_baseline.csv`, `validation/sweep/sweep_grid.csv` and")
    w("`validation/sweep/sweep_classified.csv`. Regenerate with:")
    w("")
    w("```bash")
    w("scripts/gen_domains.py > docs/DOMAINS.md")
    w("```")
    w("")
    w("`validation/check_domains_fresh.sh` fails if this file is not what the")
    w("generator currently emits. It is **not** wired into ctest — see the note at")
    w("the end of this file for whoever sets up CI in S7.")
    w("")
    w("---")
    w("")
    w("## How to read this")
    w("")
    w("Every number below is measured, on a fixed grid of 1652 real and 1780 complex")
    w("inputs per op, scored against a `__float128` (binary128) oracle. A backend's")
    w("**cap** is the most digits its format can carry; a cell reports where the op")
    w("actually reaches that cap and where it does not.")
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
    w("  dominant classification of those failures.")
    w("")
    w("**Caveat for the binary ops** (`add sub mul div pow hypot fmod remainder`")
    w("`copysign fmax fmin fdim fma`, and the complex `add sub mul div pow`): the")
    w("magnitude axis is the **first operand only**. The second is drawn")
    w("log-uniformly over a per-op window, with one point in seven a deliberately")
    w("cancelling pair, so a failure at a given |x| may be caused by the operand")
    w("paired with it rather than by x. For those rows read the `fails` count and the")
    w("classification, and treat the band as indicative. The unary rows have no such")
    w("ambiguity.")
    w("")
    w("Classifications, from `scripts/sweep_accuracy --classify`:")
    w("")
    for k in ["UNDERFLOW", "OVERFLOW", "ARG_RANGE", "CONDITIONING", "UNEXPLAINED"]:
        w("- **%s** — %s" % (k, CLASS_BLURB[k]))
    w("")
    w("`UNDERFLOW`, `OVERFLOW` and `ARG_RANGE` are *format* limits: they are properties")
    w("of 2xFP32 or 4xFP32 and no implementation can remove them. `CONDITIONING` is a")
    w("*mathematical* limit, measured by perturbing each operand by the error the")
    w("backend actually carries and re-evaluating the oracle. **`UNEXPLAINED` is")
    w("neither** — those are the points where the answer was representable and the")
    w("problem well-conditioned, and the implementation still lost the digits. They")
    w("are filed as KI-6 through KI-11 in `docs/KNOWN_ISSUES.md`.")
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

    # ---- the tables --------------------------------------------------------
    totals = collections.Counter()
    for title, kind, ops in FAMILIES:
        w("## %s (%s)" % (title, "real" if kind == "r" else "complex"))
        w("")
        w("| op | backend | cap | mean | trusted \\|%s\\| | at cap | boundary (digits) | fails | dominant class |"
          % ("x" if kind == "r" else "z"))
        w("|---|---|---:|---:|---|---:|---|---:|---|")
        for op in ops:
            for be in BACKENDS:
                c = cell(be, kind, op, grid, baseline, by_class, by_reason)
                totals.update(c["classes"])
                w("| `%s` | %s | %.2f | %.2f | %s | %.0f%% | %s | %d | %s |" % (
                    op, be, CAPS[be], c["mean"], c["band"], c["ok_pct"],
                    c["boundary"], c["n_bad"],
                    c["dominant"] + ("" if c["unexplained"] == 0
                                     else " (**%d unexplained**)" % c["unexplained"])))
        w("")

        # Per-family note on where the failures concentrate, and the specific
        # excluded regions -- branch cuts, poles, zeros of the function.
        notes = []
        for op in ops:
            for be in BACKENDS:
                c = cell(be, kind, op, grid, baseline, by_class, by_reason)
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

    # ---- totals ------------------------------------------------------------
    w("## Totals across all 252 cells")
    w("")
    w("| classification | points below 50% of cap |")
    w("|---|---:|")
    for k in ["UNDERFLOW", "OVERFLOW", "ARG_RANGE", "CONDITIONING", "UNEXPLAINED"]:
        w("| %s | %d |" % (k, totals.get(k, 0)))
    w("| **total triaged** | **%d** |" % sum(totals.values()))
    w("")
    w("Out of 428,592 scored points.")
    w("")
    w("---")
    w("")
    w("## Note for CI (S7)")
    w("")
    w("`validation/check_domains_fresh.sh` regenerates this file and diffs it against")
    w("the committed copy, exiting nonzero on any difference. It needs only Python 3")
    w("and the three committed CSVs — no build, no Kokkos, no libquadmath — and runs")
    w("in well under a second. It is deliberately **not** registered as a ctest test,")
    w("because the rest of the suite tests compiled numerics and a documentation")
    w("freshness check does not belong in the same gate. Wire it into CI directly.")
    w("")
    w("Note that the check only proves the markdown matches the CSVs. If a numeric fix")
    w("lands, the CSVs must be regenerated first — `scripts/sweep_accuracy` to")
    w("re-baseline, then `scripts/sweep_accuracy --classify")
    w("validation/sweep/sweep_classified.csv` — and only then this file.")

    sys.stdout.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
