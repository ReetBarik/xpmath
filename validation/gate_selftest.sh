#!/usr/bin/env bash
# A GATE NOBODY HAS SEEN FAIL IS A GATE NOBODY HAS TESTED.
#
# Both correctness gates are negative assertions -- "no point got worse", "no
# point above bound that is not registered". A negative assertion that can only
# be observed passing is indistinguishable from one that cannot fail at all,
# and this repository has already shipped one of those (tf_accuracy_test, whose
# exit code could not report a failure, for weeks).
#
# So: poison the input, demand the gate FAIL, and demand it PASS again on the
# clean input. The second half matters as much as the first -- a gate wired to
# fail unconditionally would satisfy every poison case here and be just as
# useless.
#
# The monotone gate was in fact blind when this was written. `bound` and
# `digits`, two of the four value columns of every baseline row, were parsed
# and discarded (`(void)dig; (void)base_bound;`), so rewriting the last numeric
# column of all 428,592 rows to zero still printed
# "unchanged: 428592 / RESULT: PASS". Case `bound` below is exactly that edit.
#
# Everything is written under the build directory. validation/ is READ ONLY
# here: the committed baseline and register are the fixtures, never the target.
#
#   validation/gate_selftest.sh <sweep_accuracy binary> monotone|absolute [workdir]
#
# ctest passes the build directory as <workdir>. Each sweep run is ~7s and each
# case is one run, so budget ~45s per mode.
set -u

bin="${1:?usage: gate_selftest.sh <sweep_accuracy> monotone|absolute [workdir]}"
mode="${2:?usage: gate_selftest.sh <sweep_accuracy> monotone|absolute [workdir]}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$root/validation/sweep/sweep_baseline.csv.gz"
reg="$root/validation/sweep/open_defects.txt"

[ -x "$bin" ]  || { echo "selftest: $bin is not executable" >&2; exit 2; }
[ -f "$base" ] || { echo "selftest: missing $base" >&2; exit 2; }
[ -f "$reg" ]  || { echo "selftest: missing $reg" >&2; exit 2; }

# Poisoned fixtures live in the build tree, never next to the real ones: a
# self-test that writes into validation/ is one interrupted run away from
# leaving a poisoned baseline where the real gate will read it.
if [ "$#" -ge 3 ]; then
  work="$3/gate_selftest.$mode"
  rm -rf "$work"; mkdir -p "$work" || exit 2
else
  work="$(mktemp -d "${TMPDIR:-/tmp}/gate_selftest.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
fi
case "$work" in
  "$root"/validation/*) echo "selftest: refusing to write inside validation/" >&2; exit 2 ;;
esac

fails=0

# run <name> <expect> <args...>
#
# <expect> is `pass`, or an EXACT EXIT CODE. "Nonzero" is not good enough: the
# gate has four distinct failure doors and a case that starts failing through
# the wrong one is a case that has stopped testing what it was written for.
#
#   1  a point got worse            2  coverage removed / unreadable baseline
#   3  record drift (bound/digits/state, or a moved oracle fingerprint)
#   5  improvement drift -- the record is stale in the BETTER direction
#
# This is the same discipline as oracle_conv_selftest.sh's why[] table, and it
# is here for the same reason: when exit 5 was added, `digits-inflated` would
# have stayed green while silently changing which door it exercised.
#
# `fail` is still accepted, meaning "any nonzero", for the absolute-mode cases
# where the code is not the point.
run() {
  local name="$1" expect="$2"; shift 2
  local out="$work/$name.log" rc
  "$bin" --quiet "$@" > "$out" 2>&1; rc=$?
  local got ok
  if [ "$rc" -eq 0 ]; then got=pass; else got="$rc"; fi
  case "$expect" in
    pass) [ "$rc" -eq 0 ] && ok=1 || ok=0 ;;
    fail) [ "$rc" -ne 0 ] && ok=1 || ok=0 ;;
    *)    [ "$rc" -eq "$expect" ] && ok=1 || ok=0 ;;
  esac
  if [ "$ok" = 1 ]; then
    printf '  ok    %-22s expected %-4s got %s (exit %d)\n' "$name" "$expect" "$got" "$rc"
    grep -E '^RESULT|^  (REGRESSION|RECORD|NEW|STALE)' "$out" | head -3 | sed 's/^/          /'
  else
    printf '  FAIL  %-22s expected %-4s got %s (exit %d)\n' "$name" "$expect" "$got" "$rc"
    tail -12 "$out" | sed 's/^/          /'
    fails=$((fails + 1))
  fi
}

# Rewrite one 1-based field of every data row of the baseline. Header comments
# and the column header are passed through untouched, so the poisoned file is
# structurally valid and parses -- the point is that a gate must notice a file
# it can read perfectly well and that is nonetheless a lie.
poison_col() {  # poison_col <field> <value> <outfile>
  gzip -dc "$base" \
    | awk -F, -v OFS=, -v c="$1" -v v="$2" \
        '/^#/ {print; next} /^backend,/ {print; next} {$c = v; print}' \
    | gzip > "$3"
}

case "$mode" in
monotone)
  echo "=== sweep_monotone_gate self-test ==============================="

  # The control. If this does not pass, nothing below means anything.
  # THIS BUILD'S FINGERPRINT, asked of the binary rather than assumed.
  #
  # Every poison below is derived from the committed baseline, which carries the
  # fingerprint of the TOOLCHAIN OF RECORD. On any machine whose libquadmath
  # differs -- every GitHub runner -- that mismatches, and the gate correctly
  # refuses to attribute improvements to the library when the reference itself
  # moved. Which means an exit-5 poison built from the committed header CANNOT
  # FIRE there: measured, `ulps-inflated` and `fake-improvement-1row` both
  # passed on the runner while passing here, because "here" is the one place the
  # fingerprint matches.
  #
  # So the poisons that are meant to be ATTRIBUTABLE are stamped with the
  # fingerprint this build actually produces. Then they mean "same reference,
  # better rows" everywhere, which is the condition exit 5 is for.
  "$bin" --quiet --out "$work/fp_probe.csv" >/dev/null 2>&1
  self_fp="$(sed -n 's/^# oracle-fingerprint: //p' "$work/fp_probe.csv" | head -1)"
  if [ -z "$self_fp" ]; then
    echo "  FAIL  could not read this build's oracle fingerprint" >&2
    fails=$((fails + 1))
    self_fp=0000000000000000
  fi
  echo "  note  this build's oracle fingerprint: $self_fp"

  # stamp <src.gz> <dst.gz> -- rewrite the fingerprint header to this build's.
  stamp() {
    zcat "$1" | sed "s/^# oracle-fingerprint: .*/# oracle-fingerprint: ${self_fp}/" \
      | gzip > "$2"
  }

  run clean pass --baseline "$base"

  # The measurement of record, zeroed. A baseline claiming every point was
  # exact must make the real sweep look catastrophically worse.
  poison_col 6 0 "$work/ulps.csv.gz"
  run ulps-zeroed 1 --baseline "$work/ulps.csv.gz"

  # THE ORIGINAL BLIND SPOT: the last numeric column is `bound`, not `ulps`.
  poison_col 7 0 "$work/bound.csv.gz"
  run bound-zeroed 3 --baseline "$work/bound.csv.gz"

  # The other discarded column.
  poison_col 5 0 "$work/digits.csv.gz"
  run digits-zeroed 3 --baseline "$work/digits.csv.gz"

  # A verdict column flipped from scored to unresolved: every point loses its
  # verdict, which is a regression even though no ulp count moved.
  poison_col 8 U "$work/state.csv.gz"
  run state-flipped 3 --baseline "$work/state.csv.gz"

  # Truncation. A short baseline must not be read as agreement on the rows it
  # happens to contain.
  gzip -dc "$base" | head -1000 | gzip > "$work/short.csv.gz"
  run truncated 2 --baseline "$work/short.csv.gz"

  # ---------------------------------------------------------------------
  # ZERO ROWS. The case `truncated` above CANNOT reach.
  #
  # `head -1000` keeps the '#' header, so declared_real/declared_cplx are set
  # and the truncation check fires. Strip the header too and every guard in
  # compare_baseline no-ops at once: declared_* stay 0 so the truncation check
  # is skipped, removed_points is 0 because there were no baseline rows to go
  # missing, and nothing counted `parsed`. The gate printed
  # "compared: 0 points ... RESULT: PASS" and exited 0 -- certifying a
  # comparison it had not made. Measured on the real binary before the fix.
  #
  # Three inputs, because they fail through different doors: an empty file, a
  # file with a header and no body, and a stream the decompressor rejects.
  # ---------------------------------------------------------------------
  : | gzip > "$work/empty.csv.gz"
  run empty-baseline 2 --baseline "$work/empty.csv.gz"

  gzip -dc "$base" | grep '^#' | gzip > "$work/header_only.csv.gz"
  run header-only 2 --baseline "$work/header_only.csv.gz"

  # Not valid gzip at all: exercises the pclose status path specifically,
  # which used to be discarded outright.
  printf 'this is not gzip data' > "$work/corrupt.csv.gz"
  run corrupt-gz 2 --baseline "$work/corrupt.csv.gz"

  # ---------------------------------------------------------------------
  # DIRECTIONAL DIGIT DRIFT.
  #
  # Digit drift no longer gates when a measured improvement explains it, so
  # that an accuracy fix can pass a parent-vs-HEAD comparison. That relaxation
  # is only safe if the OTHER direction still fails, and if the forgiving path
  # cannot be reached by a point that got worse.
  #
  # `digits-inflated` is the guard: a baseline claiming MORE digits than this
  # build derives, with the ulps column left alone. No row can be classified as
  # improved (the error did not shrink), so every one of the 428,592 rows is
  # unexplained digit drift and the gate must still exit nonzero. If someone
  # later widens the exemption to "any digit movement", this case goes green
  # and tells them.
  # ---------------------------------------------------------------------
  gzip -dc "$base" \
    | awk -F, -v OFS=, '/^#/ {print; next} /^backend,/ {print; next}
                        {$5 = $5 + 5.0; print}' \
    | gzip > "$work/digits_up.csv.gz"
  run digits-inflated 3 --baseline "$work/digits_up.csv.gz"

  # And the ulps column inflated on its own: the record says every point was
  # far WORSE than this build measures. Those rows do satisfy "error shrank",
  # which is the improvement predicate -- but the digit column is untouched, so
  # there is no digit drift to forgive and the run must still come back clean.
  # This pins the exemption to rows where BOTH moved consistently, rather than
  # letting a shrinking error excuse an arbitrary record.
  gzip -dc "$base" \
    | awk -F, -v OFS=, '/^#/ {print; next} /^backend,/ {print; next}
                        {$6 = $6 * 1000.0 + 1.0; print}' \
    | gzip > "$work/ulps_inflated.csv.gz"
  # POLARITY FLIPPED BY #9, DELIBERATELY. This case used to assert `pass`, and
  # that assertion WAS the bug: a baseline claiming every row is 1000x worse
  # than the build scored `increased: 397409` and exited 0. Measured on the
  # pre-fix binary before the change landed.
  #
  # What it originally pinned must not evaporate: it was written to stop a
  # SHRINKING error from excusing an arbitrary record through the digit
  # exemption. It still tests that -- the exemption no longer excuses anything,
  # because improvement drift now has its own door. The case is the same, the
  # expected door changed from "none" to 5.
  stamp "$work/ulps_inflated.csv.gz" "$work/ulps_inflated_fp.csv.gz"
  run ulps-inflated 5 --baseline "$work/ulps_inflated_fp.csv.gz"

  # ---------------------------------------------------------------------
  # IMPROVEMENT DRIFT (#9). The record is stale in the BETTER direction.
  #
  # `increased` and `drift_digits_up` were computed, printed and then dropped
  # before the exit code, so no amount of unexplained improvement could fail.
  # Two cases, because the whole-file version could be satisfied by some future
  # "N% of rows moved" heuristic and the single-row one cannot.
  #
  # Both exited 0 on the pre-fix binary. That was measured, not assumed --
  # a poison nobody has seen fail is not a poison.
  # ---------------------------------------------------------------------
  zcat "$base" | awk -F, -v OFS=, '
      /^#/{print;next} /^backend,/{print;next}
      !done && $8=="S" && $6+0>0 { $6=1e6; $5="14.00"; done=1; print; next }
      {print}' | gzip > "$work/fake_one.csv.gz"
  stamp "$work/fake_one.csv.gz" "$work/fake_one_fp.csv.gz"
  run fake-improvement-1row 5 --baseline "$work/fake_one_fp.csv.gz"

  # And the negative: an improvement UNDER the noise floor must stay silent, or
  # the gate fires on run-to-run wiggle. 1.05x is well inside kNoiseFactor
  # (1.2589). Digits are left alone here -- moving them would be real drift and
  # would exit 3 through a different door, which is what a first attempt at this
  # case actually did.
  zcat "$base" | awk -F, -v OFS=, '/^#/{print;next} /^backend,/{print;next}
                                   {$6 = $6*1.05; print}' \
      | gzip > "$work/subnoise.csv.gz"
  # Stamped too: otherwise on a mismatching runner this passes because the
  # improvements were suppressed, not because they were sub-noise -- a negative
  # control that is right for the wrong reason is not a control.
  stamp "$work/subnoise.csv.gz" "$work/subnoise_fp.csv.gz"
  run improvement-subnoise pass --baseline "$work/subnoise_fp.csv.gz"

  # ---------------------------------------------------------------------
  # A MISMATCHED ORACLE FINGERPRINT IS NOT A FAILURE.
  #
  # ci.yml records that the GitHub runner's libquadmath differs from the
  # toolchain of record, so the fingerprint mismatches there on EVERY run. The
  # repo has decided that is expected and non-gating: it says the reference
  # moved, not that this commit regressed.
  #
  # This case exists because nothing asserted it. Every other monotone case runs
  # on the toolchain of record, where the fingerprint matches by construction,
  # so the harness was structurally blind to the one condition CI experiences on
  # every single run. The first #9 interlock keyed on `fp_seen && !fp_ok`, went
  # 13/13 green here, and turned 436 hairline improvements into a hard CI
  # failure of both this self-test and sweep_monotone_gate.
  #
  # Simulating it costs one sed: the fingerprint is a header line.
  # ---------------------------------------------------------------------
  zcat "$base" \
    | sed 's/^# oracle-fingerprint: .*/# oracle-fingerprint: deadbeefdeadbeef/' \
    | gzip > "$work/fp_mismatch.csv.gz"
  run fingerprint-mismatch pass --baseline "$work/fp_mismatch.csv.gz"

  # And with improvements on top of the mismatch -- the exact shape the GitHub
  # runner reports on a CLEAN TREE, where a different libquadmath makes a few
  # hundred rows hairline-better than the record.
  #
  # The answer is PASS, and I asserted 5 here first and broke CI again. Exit 5
  # says "the record is stale, re-record it"; that is only true if the reference
  # is unchanged. With a mismatched fingerprint the rows are better because the
  # ORACLE moved, so there is nothing to attribute and nothing to re-record.
  # Unattributable improvement is not a finding.
  zcat "$base" \
    | sed 's/^# oracle-fingerprint: .*/# oracle-fingerprint: deadbeefdeadbeef/' \
    | awk -F, -v OFS=, '
        /^#/{print;next} /^backend,/{print;next}
        !done && $8=="S" && $6+0>0 { $6=1e6; $5="14.00"; done=1; print; next }
        {print}' | gzip > "$work/fp_mismatch_improved.csv.gz"
  run fingerprint-mismatch-improved pass --baseline "$work/fp_mismatch_improved.csv.gz"
  ;;

absolute)
  echo "=== sweep_absolute_gate self-test ==============================="

  run clean pass --ulp --register "$reg"

  # DIRECTION A -- an above-bound point that the register does not list.
  #
  # This used to be poisoned by deleting a line from the register, which only
  # works while the register still HAS a line: it borrowed a real open defect
  # as its fixture, so the day the last defect was fixed the poison would have
  # become a no-op and the case would have gone quietly green on nothing. It
  # did become a no-op -- the register is empty as of the Payne-Hanek work.
  #
  # So the above-bound point is manufactured instead, by collapsing the slack
  # multiplier the gate allows. Nothing about the library or the register is
  # borrowed, and the case keeps biting at an empty register. `slack` asserts
  # the fixture: at this allowance there must BE unlisted above-bound points,
  # or the poison is not a poison.
  slack=0.001
  "$bin" --quiet --ulp --ulp-allowance "$slack" --register "$reg" \
      > "$work/no-slack.probe" 2>&1
  grep -qE '^ *NEW \(unlisted\): *[1-9]' "$work/no-slack.probe" \
    || { echo "  FAIL  fixture: allowance $slack left no unlisted above-bound point" >&2
         fails=$((fails+1)); }
  run no-slack fail --ulp --ulp-allowance "$slack" --register "$reg"

  # DIRECTION B -- a listed point that is not above bound, i.e. a register that
  # has rotted and must shrink. Independent of how many real entries there are.
  cp "$reg" "$work/added.txt"
  echo "DD r add 999999" >> "$work/added.txt"
  run bogus-added fail --ulp --register "$work/added.txt"

  # The two register-shaped poisons below only exist while the register is
  # non-empty. They are strictly extra -- direction A is covered by `no-slack`
  # and direction B by `bogus-added` either way -- and they are SKIPPED rather
  # than run vacuously, because a poison that cannot poison must not print ok.
  if grep -qE '^[^#[:space:]]' "$reg"; then
    # A registered defect deleted: the point is still above bound, so it is now
    # an unlisted above-bound point -- a NEW defect as far as the gate knows.
    awk '/^[^#[:space:]]/ && !done {done=1; next} {print}' "$reg" > "$work/deleted.txt"
    cmp -s "$reg" "$work/deleted.txt" \
      && { echo "  FAIL  fixture: nothing was deleted from the register" >&2; fails=$((fails+1)); }
    run entry-deleted fail --ulp --register "$work/deleted.txt"

    # An empty register: every real above-bound point is unlisted. A gate that
    # reads "no exceptions listed" as "no exceptions" is the same blindness in
    # the other file.
    : > "$work/empty.txt"
    run register-emptied fail --ulp --register "$work/empty.txt"
  else
    printf '  skip  %-22s register is empty; no-slack covers this direction\n' "entry-deleted"
    printf '  skip  %-22s register is empty; no-slack covers this direction\n' "register-emptied"
  fi
  ;;

*)
  echo "selftest: unknown mode '$mode' (want monotone or absolute)" >&2; exit 2 ;;
esac

echo
if [ "$fails" -eq 0 ]; then
  echo "RESULT: PASS -- the $mode gate passes clean input and fails every poison"
  exit 0
fi
echo "RESULT: FAIL -- $fails self-test case(s) did not behave as required." >&2
echo "  A poison case that PASSES means the $mode gate cannot see that edit." >&2
exit 1
