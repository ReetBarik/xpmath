#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
"""ctest p3_tier_gate_selftest: p3_tier_gate.py on synthetic inputs.

Every row of the policy table, every pre-check, every exit-3 door and every exit
code (0, 1, 2, 3) is asserted EXACTLY, and each case also asserts a phrase from
the report so a case cannot pass through a different door than the one it is
named for. Synthetic data only: nothing here reads a real baseline.
Run: python3 -I scripts/p3_tier_gate_selftest.py
"""
import copy, gzip, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, 'p3_tier_gate.py')
LOCS = ['host', 'a100', 'b200', 'mi250', 'mi300']
SUFFIX = {'host': '', 'a100': '_a100', 'b200': '_b200', 'mi250': '_mi250', 'mi300': '_mi300'}
ARCH = {'a100': 'sm_80', 'b200': 'sm_100', 'mi250': 'gfx90a', 'mi300': 'gfx942', 'gb300': 'sm_103'}
HEADER = 'where,backend,kind,op,point,digits,ulps,bound,state\n'


def base_rows():
    rows = {}
    for be in ('DD', 'FF'):
        for kind in ('r', 'c'):
            for op in ('add', 'exp', 'log'):
                for pt in range(4):
                    ulps = 0.3 + 0.2 * pt + (0.1 if op == 'exp' else 0.0)
                    rows[(be, kind, op, str(pt))] = [ulps, 100.0, 'S']
    rows[('DD', 'r', 'add', '3')] = [2.0, 100.0, 'S']       # above 1 ulp, well inside its bound
    rows[('DD', 'r', 'exp', '3')] = [200.0, 100.0, 'S']     # a registered defect: above its bound
    rows[('DD', 'r', 'log', '1')] = [90.0, 100.0, 'S']      # near its bound
    return rows


def write_scored(path, rows, where='host'):
    with gzip.open(path, 'wt') as f:
        f.write('# xp-sweep-2 (synthetic)\n' + HEADER)
        for (be, kind, op, pt), (u, b, s) in sorted(rows.items()):
            f.write('%s,%s,%s,%s,%s,31.00,%r,%r,%s\n' % (where, be, kind, op, pt, u, b, s))


def sh(*cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise SystemExit('selftest setup failed: %s\n%s' % (' '.join(cmd), r.stderr))
    return r.stdout.strip()


class Repo:
    """A tiny git repo with the layout the gate reads; `tweak` edits Current/Reference."""
    def __init__(self, root, tweak=None):
        self.root = root
        sw = os.path.join(root, 'validation', 'sweep')
        os.makedirs(os.path.join(sw, 'reference_4d597fd'))
        os.makedirs(os.path.join(root, 'scripts'))
        self.strict = {l: base_rows() for l in LOCS}
        self.fast = {l: base_rows() for l in LOCS}
        self.ref = {l: base_rows() for l in LOCS}
        if tweak:
            tweak(self)
        for l in LOCS:
            write_scored(os.path.join(sw, 'sweep_baseline%s.csv.gz' % SUFFIX[l]), self.strict[l], l)
            write_scored(os.path.join(sw, 'sweep_baseline_fast%s.csv.gz' % SUFFIX[l]), self.fast[l], l)
            write_scored(os.path.join(sw, 'reference_4d597fd', 'sweep_baseline%s.csv.gz' % SUFFIX[l]),
                         self.ref[l], l)
        for name, text in (('sweep_accuracy.cpp', 'static const double kUlpAllowance = 8.0;\nint bound;\n'),
                           ('sweep_inputs.hpp', '// inputs\n'), ('sweep_ops.hpp', '// ops\n')):
            with open(os.path.join(root, 'scripts', name), 'w') as f:
                f.write(text)
        with open(os.path.join(sw, 'open_defects.txt'), 'w') as f:
            f.write('DD r exp 3\n')
        sh('git', 'init', '-q', cwd=root)
        sh('git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'add', '-A', cwd=root)
        sh('git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'base', cwd=root)
        self.base = sh('git', 'rev-parse', 'HEAD', cwd=root)


def default_perf():
    out = {}
    for arch in ARCH.values():
        out[arch] = [('DD_fast', 'add', 'chain', 100, 100), ('DD_fast', 'exp', 'func', 100, 90),
                     ('cDD_fast', 'exp', 'func', 100, 100), ('DD_fast', 'science', 'science', 100, 100)]
    out['sm_103'] = [('DD_fast', 'add', 'chain', 100, 300), ('DD_fast', 'exp', 'func', 100, 90)]
    return out


def write_perf(d, perf):
    os.makedirs(d, exist_ok=True)
    for arch, rows in perf.items():
        with open(os.path.join(d, 'ratios_%s.csv' % arch), 'w') as f:
            f.write('type,op,kind,before_ns,after_ns,ratio\n')
            for t, op, kind, b, a in rows:
                f.write('%s,%s,%s,%s,%s,%.6f\n' % (t, op, kind, b, a, a / b))


def run_case(repo, fresh_mut=None, perf_mut=None, exits=None, drop=(), deferred='', ops='exp',
             backend='DD', base=None, extra=None, repo_mut=None, raw_fresh=None, workdir=None):
    work = tempfile.mkdtemp(dir=workdir)
    root = repo.root
    if repo_mut:                      # edit the working tree after the base commit
        root = os.path.join(work, 'repo')
        shutil.copytree(repo.root, root)
        repo_mut(root)
    fresh = os.path.join(work, 'fresh')
    os.makedirs(fresh)
    rows = {l: copy.deepcopy(repo.fast[l]) for l in LOCS}
    if fresh_mut:
        fresh_mut(rows)
    for l in LOCS:
        if l in drop:
            continue
        write_scored(os.path.join(fresh, l + '.csv.gz'), rows[l], l)
        with open(os.path.join(fresh, l + '.exits'), 'w') as f:
            f.write('absolute=%d\nmonotone=0\n' % (exits or {}).get(l, 0))
    if raw_fresh:
        raw_fresh(fresh)
    perf = default_perf()
    if perf_mut:
        perf_mut(perf)
    write_perf(os.path.join(work, 'perf'), perf)
    cmd = [sys.executable, '-I', GATE, '--backend', backend, '--ops', ops, '--base', base or repo.base,
           '--fresh', fresh, '--perf', os.path.join(work, 'perf'), '--repo', root]
    if deferred:
        cmd += ['--deferred', deferred]
    if extra:
        cmd = extra(cmd)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    shutil.rmtree(work, ignore_errors=True)
    return r.returncode, r.stdout + r.stderr


def set_perf(typ, op, after, archs=None):
    def go(p):
        for a, rows in p.items():
            if a == 'sm_103' or (archs and a not in archs):
                continue
            p[a] = [(r[0], r[1], r[2], 100, after) if (r[0], r[1]) == (typ, op) else r for r in rows]
    return go


def rename_backend(be):
    def go(p):
        for a, rows in p.items():
            p[a] = [(r[0].replace('DD', be),) + r[1:] for r in rows]
    return go


FAILED = []
COUNT = [0]


def expect(name, code, phrase, *a, **kw):
    COUNT[0] += 1
    rc, out = run_case(*a, **kw)
    ok = rc == code and (phrase is None or phrase in out)
    print('%s %-52s exit %d (want %d)%s' % ('ok  ' if ok else 'FAIL', name, rc, code,
                                            '' if ok or rc != code else '  missing phrase %r' % phrase))
    if not ok:
        FAILED.append(name)
        print('\n'.join('      | ' + ln for ln in out.splitlines()[-12:]))


def set_row(rows, key, ulps=None, bound=None, state=None, locs=LOCS):
    for l in locs:
        r = rows[l][key]
        if ulps is not None: r[0] = ulps
        if bound is not None: r[1] = bound
        if state is not None: r[2] = state


def main():
    tmp = tempfile.mkdtemp(prefix='p3gate_selftest_')
    try:
        plain = Repo(os.path.join(tmp, 'plain'))
        K = ('DD', 'r', 'exp', '1')            # base ulps 0.7, bound 100
        KL = ('DD', 'r', 'log', '1')           # base 90, bound 100
        KA = ('DD', 'r', 'add', '3')           # base 2.0
        KD = ('DD', 'r', 'exp', '3')           # base 200 > bound 100 (registered defect)
        W = tmp

        # ---------------- exit 0: every STRICT rule passes ----------------
        expect('strict pass: identical rows, 10% faster', 0, 'accept into strict', plain, workdir=W)
        expect('strict pass: speed 0.9708 (just inside 3%)', 0, 'exit 0', plain, workdir=W,
               perf_mut=set_perf('DD_fast', 'exp', 97.08))
        expect('strict pass: only the complex row of the op is faster', 0, 'exit 0', plain, workdir=W,
               perf_mut=lambda p: (set_perf('DD_fast', 'exp', 100)(p), set_perf('cDD_fast', 'exp', 90)(p)))
        expect('strict pass: speedup on a single GPU is enough', 0, 'exit 0', plain, workdir=W,
               perf_mut=lambda p: (set_perf('DD_fast', 'exp', 100)(p), set_perf('DD_fast', 'exp', 90, ['gfx942'])(p)))
        expect('strict pass: Worse threshold is exclusive (x1.2589254)', 0, 'exit 0', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KA, ulps=2.0 * 1.2589254))
        expect('reject: just past the Worse threshold (x1.26)', 2, 'failing the strict rules: 5', plain,
               workdir=W, fresh_mut=lambda r: set_row(r, KA, ulps=2.0 * 1.26))
        expect('strict pass: > noise but <= 1 ulp is not Worse', 0, 'exit 0', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, ('DD', 'r', 'add', '0'), ulps=0.95))
        expect('strict pass: above bound but equal to Reference', 0, 'exit 0', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KD, ulps=200.0))
        expect('strict pass: improved rows are fine', 0, 'exit 0', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KL, ulps=40.0))
        expect('strict pass: b200 deferred (no b200 data needed)', 0, 'DEFERRED', plain, workdir=W,
               drop=('b200',), deferred='b200',
               perf_mut=lambda p: p.pop('sm_100'))
        expect('strict pass: slow GB300 is report-only', 0, 'REPORT ONLY', plain, workdir=W)
        expect('strict pass: backend FF, other backends equal to Current', 0, 'exit 0', plain, workdir=W,
               backend='FF', perf_mut=rename_backend('FF'))

        # ---------------- exit 1: strict fails, every FAST rule passes -----
        def drift(rp, cur=1.2, fast=1.2, ref=1.0):
            for l in LOCS:
                rp.strict[l][K][0] = cur
                rp.fast[l][K][0] = fast
                rp.ref[l][K][0] = ref
        d_mild = Repo(os.path.join(tmp, 'drift_mild'), drift)
        expect('fast only: Worse vs Reference, within 4x', 1, 'fast only', d_mild, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.5))
        d_far = Repo(os.path.join(tmp, 'drift_far'), lambda rp: drift(rp, 3.0, 3.0, 1.0))
        expect('fast only: 3.5x Reference is still <= 4x', 1, 'fast only', d_far, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=3.5))
        d_loose = Repo(os.path.join(tmp, 'drift_loose'), lambda rp: drift(rp, 1.5, 5.0, 1.5))
        expect('fast only: equals fast Current, Worse vs strict Current', 1, 'fast only', d_loose, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=5.0))
        expect('fast only: speed exactly 0.90 passes the 10% rule', 1, 'exit 1', d_mild, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.5))

        # ---------------- exit 2: reject ---------------------------------
        expect('reject: absolute gate failed on a100', 2, 'the absolute gate exited 1', plain, workdir=W,
               exits={'a100': 1})
        expect('reject: absolute gate failed on host', 2, 'host: the absolute gate', plain, workdir=W,
               exits={'host': 2})
        expect('reject: Worse vs Current (both tiers)', 2, 'failing the fast rules: 5', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KA, ulps=2.6))
        expect('reject: fast cap, 5x Reference', 2, 'exit 2', Repo(os.path.join(tmp, 'drift_cap'),
               lambda rp: drift(rp, 4.5, 4.5, 1.0)), workdir=W, fresh_mut=lambda r: set_row(r, K, ulps=5.0))
        expect('reject: bound rule (above bound, Reference not)', 2, 'exit 2', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KL, ulps=101.0))
        expect('reject: bound rule second branch, Worse vs Reference', 2, 'exit 2', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KD, ulps=300.0))
        expect('reject: one op 5% slower on mi250', 2, 'more than 3% slower on a gated GPU: 1', plain,
               workdir=W,
               perf_mut=lambda p: p['gfx90a'].__setitem__(0, ('DD_fast', 'add', 'chain', 100, 105)))
        expect('reject: complex op 6% slower', 2, 'slower on a gated GPU: 1', plain, workdir=W,
               perf_mut=set_perf('cDD_fast', 'exp', 106, ['sm_80']))
        expect('reject: science kernel 5% slower', 2, 'slower on a gated GPU: 1', plain, workdir=W,
               perf_mut=lambda p: p['gfx942'].__setitem__(3, ('DD_fast', 'science', 'science', 100, 105)))
        expect('reject: changed op only 2% faster (strict needs 3%)', 2, 'exp=0.980',
               plain, workdir=W, perf_mut=set_perf('DD_fast', 'exp', 98))
        expect('reject: fast-only accuracy but only 5% faster', 2, 'exit 2', d_mild, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.5),
               perf_mut=set_perf('DD_fast', 'exp', 95))

        def edit(rel, fn):
            def go(root):
                path = os.path.join(root, rel)
                with open(path) as f: text = f.read()
                with open(path, 'w') as f: f.write(fn(text))
            return go
        expect('reject (pre-check): register edited', 2, 'open_defects.txt changed', plain, workdir=W,
               repo_mut=edit('validation/sweep/open_defects.txt', lambda t: t + 'DD r log 9\n'))
        expect('reject (pre-check): scorer bound code edited', 2, 'sweep_accuracy.cpp changed', plain,
               workdir=W, repo_mut=edit('scripts/sweep_accuracy.cpp', lambda t: t.replace('int bound;', 'int bound2;')))
        expect('reject (pre-check): kUlpAllowance edited', 2, 'kUlpAllowance line changed', plain, workdir=W,
               repo_mut=edit('scripts/sweep_accuracy.cpp', lambda t: t.replace('8.0', '16.0')))
        expect('reject (pre-check): sweep_ops.hpp edited', 2, 'sweep_ops.hpp changed', plain, workdir=W,
               repo_mut=edit('scripts/sweep_ops.hpp', lambda t: t + '// x\n'))

        def touch_ref(root):
            p = os.path.join(root, 'validation/sweep/reference_4d597fd/sweep_baseline_a100.csv.gz')
            with open(p, 'ab') as f: f.write(b'x')
        expect('reject (pre-check): Reference edited', 2, 'reference_4d597fd changed', plain, workdir=W,
               repo_mut=touch_ref)

        def add_ref(root):
            with open(os.path.join(root, 'validation/sweep/reference_4d597fd/extra.txt'), 'w') as f: f.write('x')
        expect('reject (pre-check): file added to Reference', 2, 'untracked files', plain, workdir=W,
               repo_mut=add_ref)

        # ---------------- exit 3: every door ------------------------------
        expect('stop: fresh file missing, not deferred', 3, 'a100: no a100.csv', plain, workdir=W, drop=('a100',))
        def no_exits(fresh): os.remove(os.path.join(fresh, 'mi300.exits'))
        expect('stop: .exits file missing', 3, 'mi300.exits', plain, workdir=W, raw_fresh=no_exits)
        def bad_exits(fresh):
            with open(os.path.join(fresh, 'host.exits'), 'w') as f: f.write('absolute=zero\n')
        expect('stop: .exits file unparsable', 3, 'cannot parse', plain, workdir=W, raw_fresh=bad_exits)
        def half_exits(fresh):
            with open(os.path.join(fresh, 'host.exits'), 'w') as f: f.write('absolute=0\n')
        expect('stop: .exits lacks monotone=', 3, 'needs absolute= and monotone=', plain, workdir=W,
               raw_fresh=half_exits)
        expect('stop: a Current baseline is missing', 3, 'missing baseline', plain, workdir=W,
               repo_mut=lambda root: os.remove(os.path.join(root, 'validation/sweep/sweep_baseline_fast_b200.csv.gz')))
        expect('stop: perf file missing for a gated GPU', 3, 'ratios_gfx942.csv', plain, workdir=W,
               perf_mut=lambda p: p.pop('gfx942'))
        # a ratio column that disagrees with before/after needs a hand-written file:
        work = tempfile.mkdtemp(dir=W)
        write_perf(os.path.join(work, 'perf'), default_perf())
        with open(os.path.join(work, 'perf', 'ratios_sm_80.csv'), 'a') as f:
            f.write('DD_fast,mul,chain,100,90,1.5\n')
        COUNT[0] += 1
        fresh = os.path.join(work, 'fresh'); os.makedirs(fresh)
        for l in LOCS:
            write_scored(os.path.join(fresh, l + '.csv.gz'), plain.fast[l], l)
            with open(os.path.join(fresh, l + '.exits'), 'w') as f: f.write('absolute=0\nmonotone=0\n')
        r = subprocess.run([sys.executable, '-I', GATE, '--backend', 'DD', '--ops', 'exp', '--base', plain.base,
                            '--fresh', fresh, '--perf', os.path.join(work, 'perf'), '--repo', plain.root],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        ok = r.returncode == 3 and 'is not after/before' in r.stdout
        print('%s %-52s exit %d (want 3)' % ('ok  ' if ok else 'FAIL', 'stop: perf ratio is not after/before',
                                             r.returncode))
        if not ok:
            FAILED.append('perf ratio'); print(r.stdout[-600:])

        def drop_row(rows):
            for l in LOCS: del rows[l][('FF', 'c', 'log', '2')]
        expect('stop: a fresh row is missing (grid changed)', 3, 'row set', plain, workdir=W, fresh_mut=drop_row)
        def add_row(rows):
            for l in LOCS: rows[l][('DD', 'r', 'newop', '0')] = [0.5, 100.0, 'S']
        expect('stop: a fresh row is extra', 3, 'row set', plain, workdir=W, fresh_mut=add_row)
        expect('stop: a state changed', 3, 'state change', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, K, state='U'))
        expect('stop: a bound moved by more than 10^0.1', 3, 'the bound moved', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, K, bound=200.0))
        expect('stop: a bound moved by exactly 2x is caught, 1.2x is not', 3, 'the bound moved', plain,
               workdir=W, fresh_mut=lambda r: set_row(r, K, bound=50.0))
        expect('pass: a bound that moved only 1.2x', 0, 'exit 0', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, K, bound=120.0))
        expect('stop: a row of another backend differs from Current', 3, 'another backend differs', plain,
               workdir=W, fresh_mut=lambda r: set_row(r, ('FF', 'r', 'exp', '1'), ulps=0.71))
        expect('stop: unknown op', 3, 'no DD rows in the sweep', plain, workdir=W, ops='exp,nosuchop')
        expect('stop: unknown backend', 3, '--backend must be', plain, workdir=W, backend='XX')
        expect('stop: deferred GPU unknown', 3, 'not a deferrable GPU', plain, workdir=W, deferred='h100')
        expect('stop: host cannot be deferred', 3, 'not a deferrable GPU', plain, workdir=W, deferred='host')
        expect('stop: gb300 cannot be deferred', 3, 'not a deferrable GPU', plain, workdir=W, deferred='gb300')
        expect('stop: changed op has no perf row anywhere', 3, 'no perf row', plain, workdir=W, ops='log')
        expect('stop: base is not a commit', 3, 'is not a commit', plain, workdir=W, base='0' * 40)
        expect('stop: empty --ops', 3, '--ops is empty', plain, workdir=W, ops=',')
        expect('stop: missing required argument', 3, None, plain, workdir=W,
               extra=lambda cmd: [c for c in cmd if c not in ('--base', plain.base)])

        def corrupt(fresh):
            with open(os.path.join(fresh, 'a100.csv.gz'), 'wb') as f: f.write(b'\x1f\x8b not gzip at all')
        expect('stop: fresh file is a corrupt gzip', 3, 'cannot', plain, workdir=W, raw_fresh=corrupt)
        def plain_text(fresh):
            os.remove(os.path.join(fresh, 'host.csv.gz'))
            with open(os.path.join(fresh, 'host.csv'), 'w') as f: f.write('hello\n')
        expect('stop: fresh file has no scored-sweep header', 3, 'no scored-sweep header', plain, workdir=W,
               raw_fresh=plain_text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\n%d cases, %d failed' % (COUNT[0], len(FAILED)))
    if FAILED:
        print('FAILED: ' + ', '.join(FAILED))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
