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
    rows[('DD', 'r', 'add', '3')] = [2.0, 100.0, 'S']
    rows[('DD', 'r', 'exp', '3')] = [200.0, 100.0, 'S']
    rows[('DD', 'r', 'log', '1')] = [90.0, 100.0, 'S']
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
             backend='DD', base=None, extra=None, repo_mut=None, raw_fresh=None, workdir=None,
             host_only=False):
    work = tempfile.mkdtemp(dir=workdir)
    root = repo.root
    if repo_mut:
        root = os.path.join(work, 'repo')
        shutil.copytree(repo.root, root)
        repo_mut(root)
    fresh = os.path.join(work, 'fresh')
    os.makedirs(fresh)
    rows = {l: copy.deepcopy(repo.fast[l]) for l in LOCS}
    if fresh_mut:
        fresh_mut(rows)
    if host_only:
        l = 'host'
        write_scored(os.path.join(fresh, l + '.csv.gz'), rows[l], l)
        with open(os.path.join(fresh, l + '.exits'), 'w') as f:
            f.write('absolute=%d\nmonotone=0\n' % (exits or {}).get(l, 0))
        if raw_fresh:
            raw_fresh(fresh)
        cmd = [sys.executable, '-I', GATE, '--host-only', '--backend', backend, '--ops', ops, '--base', base or repo.base,
               '--fresh', fresh, '--repo', root]
        if extra:
            cmd = extra(cmd)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        shutil.rmtree(work, ignore_errors=True)
        return r.returncode, r.stdout + r.stderr
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
        K = ('DD', 'r', 'exp', '1')
        KL = ('DD', 'r', 'log', '1')
        KD = ('DD', 'r', 'exp', '3')
        W = tmp

        expect('strict pass: identical rows, 10% faster', 0, 'accept into strict', plain, workdir=W)
        expect('strict pass: speed 0.9708 (just inside 3%)', 0, 'exit 0', plain, workdir=W,
               perf_mut=set_perf('DD_fast', 'exp', 97.08))
        expect('strict pass: only the complex row of the op is faster', 0, 'exit 0', plain, workdir=W,
               perf_mut=lambda p: (set_perf('DD_fast', 'exp', 100)(p), set_perf('cDD_fast', 'exp', 90)(p)))
        expect('strict pass: speedup on a single GPU is enough', 0, 'exit 0', plain, workdir=W,
               perf_mut=lambda p: (set_perf('DD_fast', 'exp', 100)(p), set_perf('DD_fast', 'exp', 90, ['gfx942'])(p)))
        expect('strict pass: improved rows are fine', 0, 'exit 0', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KL, ulps=40.0))
        expect('strict pass: b200 deferred (no b200 data needed)', 0, 'DEFERRED', plain, workdir=W,
               drop=('b200',), deferred='b200',
               perf_mut=lambda p: p.pop('sm_100'))
        expect('strict pass: slow GB300 is report-only', 0, 'REPORT ONLY', plain, workdir=W)
        expect('strict pass: backend FF, other backends equal to Current', 0, 'exit 0', plain, workdir=W,
               backend='FF', perf_mut=rename_backend('FF'))

        def mk_rf_089(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 0.089
                repo.strict[l][K][0] = 0.089
                repo.fast[l][K][0] = 0.089
        r089 = Repo(os.path.join(tmp, 'r089'), mk_rf_089)
        expect('fast only: r=0.089 f=1.51 (strict fail)', 1, 'fast only', r089, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.51))
        def mk_r06(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 0.6
                repo.strict[l][K][0] = 0.6
                repo.fast[l][K][0] = 0.6
        r06 = Repo(os.path.join(tmp, 'r06'), mk_r06)
        expect('strict pass: r=0.6 f=1.0', 0, 'exit 0', r06, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.0))
        def mk_r12(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 1.2
                repo.strict[l][K][0] = 1.2
                repo.fast[l][K][0] = 1.2
        r12 = Repo(os.path.join(tmp, 'r12'), mk_r12)
        expect('fast only: strict fail r=1.2 f=1.8', 1, 'fast only', r12, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.8))
        expect('strict pass: r=1.2 f=1.7 edge', 0, 'exit 0', r12, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.7))
        def mk_r3_b90(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 3.0; repo.ref[l][K][1] = 90.0
                repo.strict[l][K][0] = 3.0; repo.strict[l][K][1] = 90.0
                repo.fast[l][K][0] = 3.0; repo.fast[l][K][1] = 90.0
        r3b90 = Repo(os.path.join(tmp, 'r3b90'), mk_r3_b90)
        expect('reject: fast fail r=3.0 f=4.2 b=90', 2, 'reject', r3b90, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=4.2, bound=90.0))
        def mk_r10_b20(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 10.0; repo.ref[l][K][1] = 20.0
                repo.strict[l][K][0] = 10.0; repo.strict[l][K][1] = 20.0
                repo.fast[l][K][0] = 10.0; repo.fast[l][K][1] = 20.0
        r10b20 = Repo(os.path.join(tmp, 'r10b20'), mk_r10_b20)
        expect('fast only: r=10 f=10.9 b=20 fast pass', 1, 'fast only', r10b20, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=10.9, bound=20.0))
        expect('reject: fast fail r=10 f=11.1 b=20', 2, 'reject', r10b20, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=11.1, bound=20.0))
        def mk_r25_b3(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 2.5; repo.ref[l][K][1] = 3.0
                repo.strict[l][K][0] = 2.5; repo.strict[l][K][1] = 3.0
                repo.fast[l][K][0] = 2.5; repo.fast[l][K][1] = 3.0
        r25b3 = Repo(os.path.join(tmp, 'r25b3'), mk_r25_b3)
        expect('reject: bound fail r=2.5 f=3.4 b=3', 2, 'reject', r25b3, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=3.4, bound=3.0))
        def mk_r5_b3(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 5.0; repo.ref[l][K][1] = 3.0
                repo.strict[l][K][0] = 5.0; repo.strict[l][K][1] = 3.0
                repo.fast[l][K][0] = 5.0; repo.fast[l][K][1] = 3.0
        r5b3 = Repo(os.path.join(tmp, 'r5b3'), mk_r5_b3)
        expect('strict pass: bound pass r=5 f=5.4 b=3', 0, 'exit 0', r5b3, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=5.4, bound=3.0))
        def mk_cur02_ref15(repo):
            for l in LOCS:
                repo.ref[l][K][0] = 1.5
                repo.strict[l][K][0] = 0.2
                repo.fast[l][K][0] = 1.5
        cur02 = Repo(os.path.join(tmp, 'cur02'), mk_cur02_ref15)
        expect('fast only: strict fail via Current only c=0.2 r=1.5 f=1.6', 1, 'fast only', cur02, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.6))
        expect('host-only strict pass', 0, 'HOST-ONLY', r06, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.0), host_only=True)
        expect('host-only fast only', 1, 'HOST-ONLY', r089, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.51), host_only=True)
        expect('host-only fast only r=1.2 f=1.8', 1, 'HOST-ONLY', r12, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.8), host_only=True)
        expect('host-only reject bound fail', 2, 'HOST-ONLY', r25b3, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=3.4, bound=3.0), host_only=True)
        def host_missing(fresh):
            os.remove(os.path.join(fresh, 'host.csv.gz'))
        expect('host-only stop: missing file', 3, 'no host.csv', r06, workdir=W, host_only=True, raw_fresh=host_missing)

        def drift(repo, cur=1.5, ref=1.0):
            for l in LOCS:
                repo.strict[l][K][0] = cur
                repo.fast[l][K][0] = cur
                repo.ref[l][K][0] = ref
        d2 = Repo(os.path.join(tmp, 'drift2'), lambda rp: drift(rp, 1.5, 1.0))
        expect('fast only: generic drifts', 1, 'fast only', d2, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.6))
        expect('fast only: speed exactly 0.90 passes the 10% rule', 1, 'exit 1', d2, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.6))

        expect('reject: absolute gate failed on a100', 2, 'the absolute gate exited 1', plain, workdir=W,
               exits={'a100': 1})
        expect('reject: absolute gate failed on host', 2, 'host: the absolute gate', plain, workdir=W,
               exits={'host': 2})
        expect('reject: bound rule (above bound, Reference not)', 2, 'reject', plain, workdir=W,
               fresh_mut=lambda r: set_row(r, KL, ulps=101.0))
        expect('reject: bound rule Registered defect still requires bound pass', 2, 'reject', plain, workdir=W,
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
        expect('reject: fast-only accuracy but only 5% faster', 2, 'reject', d2, workdir=W,
               fresh_mut=lambda r: set_row(r, K, ulps=1.6),
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
        COUNT[0] += 1
        work2 = tempfile.mkdtemp(dir=W)
        os.makedirs(os.path.join(work2, 'fresh'))
        write_scored(os.path.join(work2, 'fresh', 'host.csv.gz'), plain.fast['host'], 'host')
        with open(os.path.join(work2, 'fresh', 'host.exits'), 'w') as f: f.write('absolute=0\nmonotone=0\n')
        write_perf(os.path.join(work2, 'perf'), default_perf())
        r = subprocess.run([sys.executable, '-I', GATE, '--host-only', '--backend', 'DD', '--ops', 'exp', '--base', plain.base,
                            '--fresh', os.path.join(work2, 'fresh'), '--perf', os.path.join(work2, 'perf'), '--repo', plain.root],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        ok = r.returncode == 3 and 'does not take --perf' in r.stdout
        print('%s %-52s exit %d (want 3)' % ('ok  ' if ok else 'FAIL', 'stop: host-only does not take --perf',
                                             r.returncode))
        if not ok:
            FAILED.append('host-only perf'); print(r.stdout[-600:])
        shutil.rmtree(work2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\n%d cases, %d failed' % (COUNT[0], len(FAILED)))
    if FAILED:
        print('FAILED: ' + ', '.join(FAILED))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
