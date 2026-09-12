#!/usr/bin/env python3
# ===========================================================================
# analyze_div_lift_csv.py -- everything the div-lift diagnosis reads out of
# the two sweep CSVs, recomputed from the files rather than quoted from a
# previous run.
#
#   usage: analyze_div_lift_csv.py <baseline.csv[.gz]> <after.csv> [census.txt]
#
# ULPS (field 6, 1-indexed) is the metric of record throughout.  Field 5,
# digits, is never used for a verdict here.  The gate's absolute criterion is
# ulps > bound * kUlpAllowance with kUlpAllowance = 8.0
# (scripts/sweep_accuracy.cpp), and it applies ONLY to state S -- U/N/X rows
# carry no verdict by design (docs/CORRECTNESS.md).
#
# Sections:
#   1. moved cells      -- op x backend cells with any row changed, at three
#                          thresholds; 3.73x is the monotone gate's factor.
#   2. defect delta     -- did ANY row anywhere cross into defect?  This is the
#                          absolute gate's question and it is asked over all
#                          436,080 rows, not just the moved ones.
#   3. the regressed rows -- every row worse by >3.73x, with its state and its
#                          headroom against bound*8 before and after.
#   4. census cross-tab -- the sqrt-precedent test.  For sqrt the decisive
#                          evidence was that ops WITHOUT sqrt in their path did
#                          not move at all.  Here: a cell may move ONLY if the
#                          probe census saw the lift change a result there.
#                          pts_changed == 0 and rows moved > 0 would break the
#                          mechanism claim.
# ===========================================================================
import gzip
import sys
from collections import defaultdict

ALLOWANCE = 8.0          # sweep_accuracy.cpp kUlpAllowance
MONOTONE = 3.73          # the monotone gate's worse-by factor


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = {}
    with op(path, "rt") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            p = line.rstrip("\n").split(",")
            if len(p) != 8 or p[0] == "backend":
                continue
            # key: backend, kind, op, point ; value: ulps, bound, state
            rows[(p[0], p[1], p[2], int(p[3]))] = (float(p[5]), float(p[6]), p[7])
    return rows


def is_defect(ulps, bound, state):
    return state == "S" and ulps > bound * ALLOWANCE


def main():
    if len(sys.argv) < 3:
        print(__doc__ or "usage: analyze_div_lift_csv.py <base> <after> [census]")
        return 2
    base = load(sys.argv[1])
    after = load(sys.argv[2])
    census_path = sys.argv[3] if len(sys.argv) > 3 else None

    print("# rows: baseline %d  after %d" % (len(base), len(after)))
    common = set(base) & set(after)
    if len(common) != len(base) or len(common) != len(after):
        print("!! key sets differ: base-only %d  after-only %d"
              % (len(set(base) - set(after)), len(set(after) - set(base))))
    print()

    # -- 1. moved cells ----------------------------------------------------
    any_ch = defaultdict(int)
    pct1 = defaultdict(int)
    reg = defaultdict(int)        # worse by more than MONOTONE
    imp = defaultdict(int)        # better by more than MONOTONE
    regressed_rows = []
    for k in sorted(common):
        u0, b0, s0 = base[k]
        u1, b1, s1 = after[k]
        if u0 == u1:
            continue
        cell = (k[0], k[1], k[2])
        any_ch[cell] += 1
        if u0 == 0.0 or abs(u1 - u0) > 0.01 * abs(u0):
            pct1[cell] += 1
        if u1 > u0 * MONOTONE or (u0 == 0.0 and u1 > 0.0):
            reg[cell] += 1
            regressed_rows.append(k)
        elif u0 > u1 * MONOTONE or (u1 == 0.0 and u0 > 0.0):
            imp[cell] += 1

    print("== 1. moved op x backend cells (ulps, field 6) ==")
    print("%-22s %8s %8s %8s %8s" % ("cell", "changed", ">1%", "worse", "better"))
    tot_r = tot_i = 0
    for cell in sorted(any_ch):
        print("%-22s %8d %8d %8d %8d"
              % (" ".join(cell), any_ch[cell], pct1[cell], reg[cell], imp[cell]))
        tot_r += reg[cell]
        tot_i += imp[cell]
    print("%-22s %8d %8d %8d %8d"
          % ("TOTAL (%d cells)" % len(any_ch), sum(any_ch.values()),
             sum(pct1.values()), tot_r, tot_i))
    print("   'worse'/'better' = ulps moved by more than %.2fx (the monotone "
          "gate's factor)" % MONOTONE)
    print()

    # -- 2. defect delta ---------------------------------------------------
    print("== 2. defect delta over ALL %d rows (state S, ulps > bound * %.1f) =="
          % (len(common), ALLOWANCE))
    d0 = {k for k in common if is_defect(*base[k])}
    d1 = {k for k in common if is_defect(*after[k])}
    print("   defects before %d   after %d" % (len(d0), len(d1)))
    became = sorted(d1 - d0)
    healed = sorted(d0 - d1)
    print("   BECAME a defect: %d" % len(became))
    for k in became[:40]:
        print("      %s %s %s %d  ulps %.6g -> %.6g  bound %.6g"
              % (k[0], k[1], k[2], k[3], base[k][0], after[k][0], after[k][1]))
    print("   stopped being a defect: %d" % len(healed))
    for k in healed[:40]:
        print("      %s %s %s %d  ulps %.6g -> %.6g  bound %.6g"
              % (k[0], k[1], k[2], k[3], base[k][0], after[k][0], after[k][1]))
    print()

    # -- 3. the regressed rows --------------------------------------------
    print("== 3. every row worse by more than %.2fx (%d rows) =="
          % (MONOTONE, len(regressed_rows)))
    print("%-3s %-2s %-6s %-6s %-4s %-4s %13s %13s %12s %8s %8s"
          % ("be", "k", "op", "point", "st0", "st1", "ulps before", "ulps after",
             "bound", "gate0", "gate1"))
    st_count = defaultdict(int)
    for k in regressed_rows:
        u0, b0, s0 = base[k]
        u1, b1, s1 = after[k]
        g0 = "-" if s0 != "S" else ("DEFECT" if u0 > b0 * ALLOWANCE else "pass")
        g1 = "-" if s1 != "S" else ("DEFECT" if u1 > b1 * ALLOWANCE else "pass")
        st_count[(s0, s1)] += 1
        print("%-3s %-2s %-6s %-6d %-4s %-4s %13.6g %13.6g %12.6g %8s %8s"
              % (k[0], k[1], k[2], k[3], s0, s1, u0, u1, b1, g0, g1))
    print("   state (before,after) histogram: %s"
          % ", ".join("%s->%s: %d" % (a, b, n) for (a, b), n in sorted(st_count.items())))
    print("   rows carrying a verdict (state S after): %d of %d"
          % (sum(n for (a, b), n in st_count.items() if b == "S"), len(regressed_rows)))
    print()

    # -- 4. census cross-tab ----------------------------------------------
    if census_path:
        cens = {}
        with open(census_path) as f:
            for line in f:
                if not line.startswith("CENSUS "):
                    continue
                p = line.split()
                # CENSUS <be> <kind> <op> calls N fires N pts_fired N pts_changed N
                cens[(p[1], p[2], p[3])] = {
                    "calls": int(p[5]), "fires": int(p[7]),
                    "pts_fired": int(p[9]), "pts_changed": int(p[11]),
                }
        print("== 4. census cross-tab: the sqrt-precedent test ==")
        print("   a cell may move rows ONLY if the probe saw the lift change a")
        print("   result there.  Violations are listed; none is the claim.")
        print("%-22s %8s %8s %10s %12s %8s"
              % ("cell", "calls", "fires", "pts_fired", "pts_changed", "moved"))
        viol_a = []     # moved but census says nothing changed
        viol_b = []     # census says changed but nothing moved (allowed, noted)
        for cell in sorted(cens):
            moved = any_ch.get(cell, 0)
            c = cens[cell]
            if moved == 0 and c["pts_changed"] == 0:
                continue
            print("%-22s %8d %8d %10d %12d %8d"
                  % (" ".join(cell), c["calls"], c["fires"],
                     c["pts_fired"], c["pts_changed"], moved))
            if moved > 0 and c["pts_changed"] == 0:
                viol_a.append(cell)
            if moved == 0 and c["pts_changed"] > 0:
                viol_b.append(cell)
        # cells the CSV says moved that the census never covered at all
        uncovered = [c for c in any_ch if c[1] == "c" and c not in cens]
        n_zero_zero = sum(1 for cell in cens
                          if cens[cell]["pts_changed"] == 0 and any_ch.get(cell, 0) == 0)
        print()
        print("   complex cells with pts_changed==0 AND 0 rows moved: %d of %d"
              % (n_zero_zero, len(cens)))
        print("   VIOLATIONS (rows moved, census saw no change): %d %s"
              % (len(viol_a), viol_a))
        print("   changed-but-unmoved (allowed: change below the ulps print): %d %s"
              % (len(viol_b), viol_b))
        print("   moved complex cells not covered by the census: %d %s"
              % (len(uncovered), uncovered))
        # zero-divide-call cells must be bit-identical
        nodiv = [cell for cell in cens if cens[cell]["calls"] == 0]
        nodiv_moved = [cell for cell in nodiv if any_ch.get(cell, 0) > 0]
        print("   cells making ZERO divide calls: %d; of those, moved: %d %s"
              % (len(nodiv), len(nodiv_moved), nodiv_moved))
        # real side: not covered by the complex census, reported separately
        real_moved = sorted(c for c in any_ch if c[1] == "r")
        print("   real-side moved cells (census is complex-only): %s"
              % ", ".join(" ".join(c) for c in real_moved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
