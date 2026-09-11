#!/usr/bin/env python3
"""Condition-number-restricted ulps report for real sin/cos/tan sweep rows.

Raw max ulps over the whole grid is not a usable discriminator between two
implementations of sin/cos: the grid deliberately includes points where the
result is near a zero of the function, and there the RELATIVE error diverges
for reasons that have nothing to do with the series.  A change can look worse
on raw max purely because it moved one near-zero cell.

The metric of record for a *decision* is therefore the worst ulps restricted
to well-conditioned points, kappa <= 4:

    sin:  kappa = |x / tan x|
    cos:  kappa = |x * tan x|
    tan:  kappa = |2x / sin 2x|      (= |x (1+tan^2 x) / tan x|)

kappa is evaluated in mpmath at 400 bits so that the restriction itself is not
polluted by the very cancellation it exists to exclude.

Usage:
    trig_kappa_report.py GRID.csv CSV [CSV ...]
    trig_kappa_report.py --top BE GRID.csv CSV      worst rows for one backend,
                                                    unrestricted, with kappa and
                                                    state shown

GRID.csv is the manifest written by `sweep_accuracy --grid-out`; it maps the
baseline's integer `point` column to the input x.  Only state=='S' rows carry a
verdict, so only those are counted (docs/CORRECTNESS.md).
"""
import csv
import sys
from collections import defaultdict

from mpmath import mp, mpf, tan as mtan, sin as msin, fabs as mfabs

mp.prec = 400

OPS = ('sin', 'cos', 'tan')
KAPPA_CUT = 4.0


def load_grid(path):
    """point index -> x, for real rows."""
    xs = {}
    with open(path) as f:
        rows = [l for l in f if not l.startswith('#')]
    for r in csv.DictReader(rows):
        if r['kind'] != 'r':
            continue
        xs[int(r['point'])] = float(r['re'])
    return xs


def kappa(op, x):
    if x == 0.0:
        return float('inf')
    xm = mpf(x)
    try:
        if op == 'sin':
            k = mfabs(xm / mtan(xm))
        elif op == 'cos':
            k = mfabs(xm * mtan(xm))
        else:
            k = mfabs(2 * xm / msin(2 * xm))
    except (ZeroDivisionError, ValueError):
        return float('inf')
    return float(k)


def load(path, xs, kcache):
    """(backend, op) -> list of (ulps, x) for scored, well-conditioned rows."""
    out = defaultdict(list)
    with open(path) as f:
        rows = [l for l in f if not l.startswith('#')]
    for r in csv.DictReader(rows):
        if r['kind'] != 'r' or r['op'] not in OPS or r['state'] != 'S':
            continue
        p = int(r['point'])
        x = xs.get(p)
        if x is None:
            continue
        key = (r['op'], p)
        if key not in kcache:
            kcache[key] = kappa(r['op'], x)
        if kcache[key] > KAPPA_CUT:
            continue
        out[(r['backend'], r['op'])].append((float(r['ulps']), x))
    return out


def top(be, grid, path, n=20):
    """Worst rows for one backend, UNRESTRICTED, annotated with kappa/state.

    Deliberately does not filter on state: a cell the sweep cannot score is
    exactly the thing that has to be checked by hand, so U/N/X rows are shown
    and labelled rather than dropped.
    """
    xs = load_grid(grid)
    rows = []
    with open(path) as f:
        src = [l for l in f if not l.startswith('#')]
    for r in csv.DictReader(src):
        if r['kind'] != 'r' or r['op'] not in OPS or r['backend'] != be:
            continue
        p = int(r['point'])
        x = xs.get(p)
        if x is None:
            continue
        rows.append((float(r['ulps']), r['op'], x, r['state'], p))
    rows.sort(reverse=True)
    print(f'{be}: worst {n} real trig rows in {path.split("/")[-1]} '
          f'(ALL states shown)\n')
    print(f'{"ulps":>12s}  {"op":4s} {"st":2s} {"kappa":>12s}  {"point":>6s}  x')
    for u, op, x, st, p in rows[:n]:
        k = kappa(op, x)
        print(f'{u:12.4f}  {op:4s} {st:2s} {k:12.4g}  {p:6d}  {x!r}')


def main():
    if sys.argv[1] == '--top':
        top(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    grid, paths = sys.argv[1], sys.argv[2:]
    xs = load_grid(grid)
    kcache = {}
    runs = [(p, load(p, xs, kcache)) for p in paths]

    print(f'worst ulps at kappa <= {KAPPA_CUT:g}, scored real rows only\n')
    hdr = f'{"be":4s}{"op":5s}' + ''.join(f'{p.split("/")[-1]:>26s}' for p in paths)
    print(hdr)
    print('-' * len(hdr))
    for be in ('DD', 'FF', 'TF', 'QF'):
        for op in OPS:
            cells = []
            for _, d in runs:
                v = d.get((be, op))
                cells.append(max(v) if v else None)
            if all(c is None for c in cells):
                continue
            line = f'{be:4s}{op:5s}'
            for c in cells:
                # x printed with repr, not %g: the worst cells sit on points
                # like 0.99999999999999978 that %g renders as "1", and chasing
                # the wrong argument in the probe wastes a whole cycle.
                line += '  ' + ('     --  ' if c is None
                                else f'{c[0]:9.4f} @ x={c[1]!r:<24s}')
            print(line)
        print()


if __name__ == '__main__':
    main()
