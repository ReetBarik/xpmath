#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
"""p3_tier_gate.py -- the mechanical accept/reject step of docs/PERF_ACCURACY_POLICY.md.

Run it with `python3 -I scripts/p3_tier_gate.py ...` (stdlib only). It judges the
candidate's scored sweep files against the policy table; it measures nothing and
issues no verdict of its own beyond that table.

  --backend DD|FF|QF|TF     the backend the task changed (each [b] is judged alone)
  --ops exp,log,...         every op the change touches, chained callers included
  --base <sha>              the task's base commit (what must not have moved)
  --fresh <dir>             <loc>.csv[.gz] + <loc>.exits for loc in host a100 b200
                            mi250 mi300 (gated) and gb300 (report only). The files
                            are the candidate's FAST-tier scored sweeps; <loc>.exits
                            holds the lines "absolute=<n>" and "monotone=<n>" (the
                            run_<arch>_sweep.sh scorer exits).
  --perf <dir>              ratios_{sm_80,sm_100,gfx90a,gfx942,sm_103}.csv from
                            perf/perf_ratios.py: type,op,kind,before_ns,after_ns,ratio
  --deferred a100,b200,...  GPUs the user declared unavailable (ledger rows owed)
  --repo <dir>              the xpmath worktree (default: .)
  --host-only               accuracy pre-screen at host only: needs only --backend
                            --ops --base --fresh (host.csv[.gz]+host.exits); no
                            --perf, no GPUs; exit 0=strict pass 1=fast pass 2=reject 3=stop

Exit codes (normal mode):
  0  every STRICT rule passed             -> accept into strict (promote, re-run TIER=both)
  1  strict failed, every FAST rule passed -> accept into fast only
  2  reject (a rule failed, a pre-check failed or the absolute gate failed)
  3  tool error / missing data / drift     -> stop and ask the user

Exit codes (host-only mode):
  0  strict accuracy passes at host
  1  strict fails but fast accuracy passes at host
  2  neither passes (or absolute gate failed, pre-check failed)
  3  tool error / missing data / drift
"""
import argparse, csv, gzip, math, os, subprocess, sys

STRICT_SLACK = 0.5
FAST_CAP = 4.0
FAST_SLACK = 1.0

STRICT_SPEED = 1.0 / 1.03    # changed op: at least 3% faster
FAST_SPEED = 1.0 / 1.10      # changed op: at least 10% faster
SLOWER_LIMIT = 1.03          # every op: no more than 3% slower

GATED = ['host', 'a100', 'b200', 'mi250', 'mi300']
DEFERRABLE = ['a100', 'b200', 'mi250', 'mi300']          # host is never deferrable
PERF_ARCH = {'a100': 'sm_80', 'b200': 'sm_100', 'mi250': 'gfx90a',
             'mi300': 'gfx942', 'gb300': 'sm_103'}
BASELINE_SUFFIX = {'host': '', 'a100': '_a100', 'b200': '_b200',
                   'mi250': '_mi250', 'mi300': '_mi300'}
FROZEN_PATHS = ['validation/sweep/open_defects.txt', 'validation/sweep/reference_4d597fd',
                'scripts/sweep_accuracy.cpp', 'scripts/sweep_inputs.hpp', 'scripts/sweep_ops.hpp']
BACKENDS = ['DD', 'FF', 'QF', 'TF']


class Stop(Exception):
    """exit 3: a tool error, missing data or drift; the user decides."""


def opener(path):
    with open(path, 'rb') as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':
        return gzip.open(path, 'rt', newline='')
    return open(path, 'r', newline='')


def find(base):
    for ext in ('.csv.gz', '.csv'):
        if os.path.isfile(base + ext):
            return base + ext
    return None


def load_scored(path):
    """{(backend,kind,op,point): (ulps_text, ulps, bound, state)} from a scored sweep."""
    rows = {}
    try:
        with opener(path) as f:
            lines = (ln for ln in f if not ln.startswith('#'))
            reader = csv.reader(lines)
            header = next(reader, None)
            if not header or 'backend' not in header or 'ulps' not in header:
                raise Stop('%s: no scored-sweep header' % path)
            ix = {n: header.index(n) for n in ('backend', 'kind', 'op', 'point', 'ulps', 'bound', 'state')
                  if n in header}
            if len(ix) != 7:
                raise Stop('%s: header lacks a required column' % path)
            for r in reader:
                if len(r) <= max(ix.values()):
                    raise Stop('%s: short row %r' % (path, r))
                key = (r[ix['backend']], r[ix['kind']], r[ix['op']], r[ix['point']])
                if key in rows:
                    raise Stop('%s: duplicate row %r' % (path, key))
                try:
                    rows[key] = (r[ix['ulps']], float(r[ix['ulps']]), float(r[ix['bound']]),
                                 r[ix['state']])
                except ValueError:
                    raise Stop('%s: non-numeric ulps/bound in %r' % (path, r))
    except (OSError, EOFError) as e:
        raise Stop('cannot read %s: %s' % (path, e))
    if not rows:
        raise Stop('%s: no rows' % path)
    return rows


def read_exits(path):
    vals = {}
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    k, _, v = ln.partition('=')
                    vals[k.strip()] = int(v)
    except (OSError, ValueError) as e:
        raise Stop('cannot parse %s: %s' % (path, e))
    if 'absolute' not in vals or 'monotone' not in vals:
        raise Stop('%s: needs absolute= and monotone= lines' % path)
    return vals


def git(repo, *args):
    try:
        return subprocess.run(['git', '-C', repo] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except OSError as e:
        raise Stop('cannot run git: %s' % e)


def prechecks(repo, base):
    """Returns a list of failure strings (exit 2 if non-empty)."""
    if git(repo, 'rev-parse', '--verify', '--quiet', base + '^{commit}').returncode != 0:
        raise Stop('--base %s is not a commit in %s' % (base, repo))
    bad = []
    for p in FROZEN_PATHS:
        if not os.path.exists(os.path.join(repo, p)):
            raise Stop('pre-check path missing: %s' % p)
        d = git(repo, 'diff', '--quiet', base, '--', p)
        if d.returncode not in (0, 1):
            raise Stop('git diff failed for %s: %s' % (p, d.stderr.strip()))
        if d.returncode == 1:
            bad.append('%s changed since %s' % (p, base[:12]))
        u = git(repo, 'ls-files', '--others', '--exclude-standard', '--', p)
        if u.stdout.strip():
            bad.append('%s has untracked files' % p)
    def ulp_lines(text):
        return [ln.strip() for ln in text.splitlines() if 'kUlpAllowance' in ln and '=' in ln]
    old = git(repo, 'show', '%s:scripts/sweep_accuracy.cpp' % base)
    if old.returncode != 0:
        raise Stop('cannot read scripts/sweep_accuracy.cpp at %s' % base)
    with open(os.path.join(repo, 'scripts/sweep_accuracy.cpp')) as f:
        new = f.read()
    if not ulp_lines(old.stdout):
        raise Stop('no kUlpAllowance line in the base scorer')
    if ulp_lines(old.stdout) != ulp_lines(new):
        bad.append('the kUlpAllowance line changed')
    return bad


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float('nan')
    i = min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1)
    return sorted_vals[max(i, 0)]


def judge_location(loc, repo, fresh_path, backend, ops, report):
    """Per-row rules at one gated location. Returns (strict_fail, fast_fail) row lists."""
    sw = os.path.join(repo, 'validation', 'sweep')
    suf = BASELINE_SUFFIX[loc]
    cur_s_p = os.path.join(sw, 'sweep_baseline%s.csv.gz' % suf)
    cur_f_p = os.path.join(sw, 'sweep_baseline_fast%s.csv.gz' % suf)
    ref_p = os.path.join(sw, 'reference_4d597fd', 'sweep_baseline%s.csv.gz' % suf)
    for p in (cur_s_p, cur_f_p, ref_p):
        if not os.path.isfile(p):
            raise Stop('%s: missing baseline %s' % (loc, p))
    fresh, cur_s, cur_f, ref = (load_scored(p) for p in (fresh_path, cur_s_p, cur_f_p, ref_p))
    if set(fresh) != set(cur_s) or set(fresh) != set(cur_f) or set(fresh) != set(ref):
        raise Stop('%s: the row set of the fresh file and the baselines differ '
                   '(fresh %d, strict %d, fast %d, reference %d rows)'
                   % (loc, len(fresh), len(cur_s), len(cur_f), len(ref)))
    strict_fail, fast_fail = [], []
    stats = {}
    seen_ops = set()
    for key, (ftxt, f, fb, fstate) in fresh.items():
        be, kind, op, pt = key
        _, cs, csb, cstate = cur_s[key]
        cftxt, cf, cfb, cfstate = cur_f[key]
        _, r, rb, rstate = ref[key]
        if fstate != cstate or fstate != cfstate:
            raise Stop('%s: state change at %s (fresh %s, strict Current %s, fast Current %s)'
                       % (loc, key, fstate, cstate, cfstate))
        if be == backend:
            seen_ops.add(op)
        if fstate == 'S':
            for b0 in (csb, cfb):
                if (fb > 0) != (b0 > 0) or (fb > 0 and abs(math.log10(fb / b0)) > 0.1 + 1e-12):
                    raise Stop('%s: the bound moved at %s (%g -> %g)' % (loc, key, b0, fb))
        if be != backend:
            if ftxt != cftxt:
                raise Stop('%s: a row of another backend differs from Current at %s (%s -> %s)'
                           % (loc, key, cftxt, ftxt))
            continue
        if fstate != 'S':
            continue
        # --- new ulp rules ---
        ok_bound = (f <= fb) or (r > rb)
        strict_ref_ok = f <= max(1.0, r + STRICT_SLACK)
        strict_cur_ok = f <= max(1.0, cs + STRICT_SLACK)
        fast_ref_ok = f <= max(FAST_CAP, r + FAST_SLACK)
        strict_ok = ok_bound and strict_ref_ok and strict_cur_ok
        fast_ok = ok_bound and fast_ref_ok
        if not strict_ok:
            strict_fail.append((loc, key, f, cs, r, fb))
        if not fast_ok:
            fast_fail.append((loc, key, f, cf, r, fb))
        if op in ops:
            st = stats.setdefault((kind, op), dict(n=0, cur=[], fresh=[],
                                                   bound_fail=0, strict_ref_fail=0, strict_cur_fail=0, fast_ref_fail=0,
                                                   worst_ratio=0.0, worst_bits=0.0))
            st['n'] += 1
            st['cur'].append(cs); st['fresh'].append(f)
            if not ok_bound:
                st['bound_fail'] += 1
            if not strict_ref_ok:
                st['strict_ref_fail'] += 1
            if not strict_cur_ok:
                st['strict_cur_fail'] += 1
            if not fast_ref_ok:
                st['fast_ref_fail'] += 1
            if fb > 0:
                st['worst_ratio'] = max(st['worst_ratio'], f / fb)
            if f > 0 and r > 0:
                st['worst_bits'] = max(st['worst_bits'], math.log2(f / r))
    for op in ops:
        if op not in seen_ops:
            raise Stop('%s: op %s has no %s rows in the sweep' % (loc, op, backend))
    for (kind, op), st in sorted(stats.items()):
        c, fr = sorted(st['cur']), sorted(st['fresh'])
        report.append('  %-5s %s %-9s n=%-5d ulps max/med/p99 %.4g/%.4g/%.4g -> %.4g/%.4g/%.4g | '
                      'fails bound %d, strict-Ref %d, strict-Cur %d, fast-Ref %d | worst ulps/bound %.4g | '
                      'worst bits lost vs Ref %.3f'
                      % (loc, 'real' if kind == 'r' else 'cplx', op, st['n'],
                         c[-1], c[len(c) // 2], percentile(c, 0.99),
                         fr[-1], fr[len(fr) // 2], percentile(fr, 0.99),
                         st['bound_fail'], st['strict_ref_fail'], st['strict_cur_fail'], st['fast_ref_fail'],
                         st['worst_ratio'], st['worst_bits']))
    return strict_fail, fast_fail


def load_ratios(path, perf_types):
    rows = []
    try:
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            need = {'type', 'op', 'kind', 'before_ns', 'after_ns', 'ratio'}
            if not rd.fieldnames or not need <= set(rd.fieldnames):
                raise Stop('%s: header must contain %s' % (path, sorted(need)))
            for r in rd:
                if r['type'] not in perf_types:
                    continue
                try:
                    b, a, q = float(r['before_ns']), float(r['after_ns']), float(r['ratio'])
                except ValueError:
                    raise Stop('%s: non-numeric row %r' % (path, r))
                if b <= 0 or abs(q - a / b) > 0.01 * max(q, a / b):
                    raise Stop('%s: ratio %g is not after/before (%g/%g) for %s/%s'
                               % (path, q, a, b, r['type'], r['op']))
                rows.append((r['type'], r['op'], r['kind'], q))
    except OSError as e:
        raise Stop('cannot read %s: %s' % (path, e))
    return rows


def main(argv):
    ap = argparse.ArgumentParser(add_help=True, description='P3 tier accept/reject gate')
    ap.add_argument('--backend', required=True)
    ap.add_argument('--ops', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--fresh', required=True)
    ap.add_argument('--perf', required=False, default=None)
    ap.add_argument('--deferred', default='')
    ap.add_argument('--repo', default='.')
    ap.add_argument('--host-only', action='store_true', dest='host_only')
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 3 if e.code else 0
    try:
        return run(args)
    except Stop as e:
        print('STOP (exit 3): %s' % e)
        return 3


def run(args):
    backend = args.backend
    if backend not in BACKENDS:
        raise Stop('--backend must be one of %s' % BACKENDS)
    ops = [o for o in args.ops.split(',') if o]
    if not ops:
        raise Stop('--ops is empty')
    deferred = [d for d in args.deferred.split(',') if d]
    for d in deferred:
        if d not in DEFERRABLE:
            raise Stop('--deferred %s: not a deferrable GPU (%s); host is never deferred and '
                       'gb300 is report-only' % (d, DEFERRABLE))
    if args.host_only:
        if args.deferred:
            raise Stop('--host-only cannot be combined with --deferred')
        if args.perf is not None:
            raise Stop('--host-only does not take --perf')
        gated = ['host']
    else:
        if args.perf is None:
            raise Stop('--perf is required (omit only with --host-only)')
        gated = [l for l in GATED if l not in deferred]
    perf_types = {backend + '_fast', 'c' + backend + '_fast'}
    report = ['P3 tier gate: backend %s, ops %s, base %s' % (backend, ','.join(ops), args.base[:12])]

    # 1. pre-checks (exit 2)
    failures = ['pre-check: ' + b for b in prechecks(args.repo, args.base)]
    if failures:
        print('\n'.join(report + failures))
        if args.host_only:
            print('RESULT: reject (exit 2) HOST-ONLY: accuracy pre-screen, no speed verdict')
        else:
            print('RESULT: reject (exit 2)')
        return 2
    report.append('pre-checks: register, Reference, scorer and kUlpAllowance unchanged since base')

    # 2. absolute gate at every gated location (exit 2)
    fresh_files = {}
    for loc in gated:
        p = find(os.path.join(args.fresh, loc))
        if p is None:
            raise Stop('%s: no %s.csv[.gz] in %s and it is not deferred' % (loc, loc, args.fresh))
        fresh_files[loc] = p
        ex = read_exits(os.path.join(args.fresh, loc + '.exits'))
        report.append('%-6s scorer exits: absolute=%d monotone=%d (monotone is informational)'
                      % (loc, ex['absolute'], ex['monotone']))
        if ex['absolute'] != 0:
            failures.append('%s: the absolute gate exited %d' % (loc, ex['absolute']))
    for d in deferred:
        report.append('%-6s DEFERRED: a ledger row is owed (docs/GPU_PERF_DEFERRED.md)' % d)

    # 3. accuracy rules per row
    strict_fail, fast_fail = [], []
    for loc in gated:
        sf, ff = judge_location(loc, args.repo, fresh_files[loc], backend, ops, report)
        strict_fail += sf
        fast_fail += ff
    for name, lst in (('strict', strict_fail), ('fast', fast_fail)):
        report.append('accuracy rows failing the %s rules: %d' % (name, len(lst)))
        for loc, key, f, cur, r, fb in lst[:3]:
            report.append('   %s %s fresh %.6g, Current %.6g, Reference %.6g, bound %.6g'
                          % (loc, '/'.join(key), f, cur, r, fb))

    if args.host_only:
        # no speed section
        for f in failures:
            report.append('FAIL: ' + f)
        print('\n'.join(report))
        if not failures and not strict_fail:
            print('RESULT: strict pass -> accept into strict (exit 0) HOST-ONLY: accuracy pre-screen, no speed verdict')
            return 0
        if not failures and not fast_fail:
            print('RESULT: strict failed, fast passed -> accept into fast only (exit 1) HOST-ONLY: accuracy pre-screen, no speed verdict')
            return 1
        print('RESULT: reject (exit 2) HOST-ONLY: accuracy pre-screen, no speed verdict')
        return 2

    # 4. speed
    slower, best = [], {op: None for op in ops}
    for loc in gated + ['gb300']:
        if loc == 'host':
            continue
        arch = PERF_ARCH[loc]
        p = os.path.join(args.perf, 'ratios_%s.csv' % arch)
        if not os.path.isfile(p):
            if loc == 'gb300':
                report.append('gb300  speed: no ratios file (report-only)')
                continue
            raise Stop('%s: no %s and it is not deferred' % (loc, p))
        rows = load_ratios(p, perf_types)
        tag = '' if loc != 'gb300' else ' (REPORT ONLY, never gated)'
        for op in ops:
            vals = [q for t, o, k, q in rows if o == op]
            if vals:
                report.append('%-6s speed %-9s ratio(after/before) %s%s'
                              % (loc, op, ' '.join('%.3f' % q for q in vals), tag))
        if loc == 'gb300':
            continue
        for t, o, k, q in rows:
            if q > SLOWER_LIMIT:
                slower.append((loc, t, o, q))
            if o in best and (best[o] is None or q < best[o]):
                best[o] = q
    for op in ops:
        if best[op] is None:
            raise Stop('changed op %s has no perf row on any gated GPU' % op)
    report.append('ops more than 3%% slower on a gated GPU: %d' % len(slower))
    for loc, t, o, q in slower[:3]:
        report.append('   %s %s/%s ratio %.3f' % (loc, t, o, q))
    strict_speed = not slower and all(best[o] <= STRICT_SPEED for o in ops)
    fast_speed = not slower and all(best[o] <= FAST_SPEED for o in ops)

    # 5. decision
    strict_ok = not failures and not strict_fail and strict_speed
    fast_ok = not failures and not fast_fail and fast_speed
    for f in failures:
        report.append('FAIL: ' + f)
    report.append('speed: best ratio per changed op %s; strict needs <= %.4f, fast needs <= %.4f'
                  % (', '.join('%s=%.3f' % (o, best[o]) for o in ops), STRICT_SPEED, FAST_SPEED))
    print('\n'.join(report))
    if strict_ok:
        print('RESULT: strict pass -> accept into strict (exit 0)')
        return 0
    if fast_ok:
        print('RESULT: strict failed, fast passed -> accept into fast only (exit 1)')
        return 1
    print('RESULT: reject (exit 2)')
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
