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

# run <name> <expect: pass|fail> <args...>
run() {
  local name="$1" expect="$2"; shift 2
  local out="$work/$name.log" rc
  "$bin" --quiet "$@" > "$out" 2>&1; rc=$?
  local got; if [ "$rc" -eq 0 ]; then got=pass; else got=fail; fi
  if [ "$got" = "$expect" ]; then
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
  run clean pass --baseline "$base"

  # The measurement of record, zeroed. A baseline claiming every point was
  # exact must make the real sweep look catastrophically worse.
  poison_col 6 0 "$work/ulps.csv.gz"
  run ulps-zeroed fail --baseline "$work/ulps.csv.gz"

  # THE ORIGINAL BLIND SPOT: the last numeric column is `bound`, not `ulps`.
  poison_col 7 0 "$work/bound.csv.gz"
  run bound-zeroed fail --baseline "$work/bound.csv.gz"

  # The other discarded column.
  poison_col 5 0 "$work/digits.csv.gz"
  run digits-zeroed fail --baseline "$work/digits.csv.gz"

  # A verdict column flipped from scored to unresolved: every point loses its
  # verdict, which is a regression even though no ulp count moved.
  poison_col 8 U "$work/state.csv.gz"
  run state-flipped fail --baseline "$work/state.csv.gz"

  # Truncation. A short baseline must not be read as agreement on the rows it
  # happens to contain.
  gzip -dc "$base" | head -1000 | gzip > "$work/short.csv.gz"
  run truncated fail --baseline "$work/short.csv.gz"

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
  run empty-baseline fail --baseline "$work/empty.csv.gz"

  gzip -dc "$base" | grep '^#' | gzip > "$work/header_only.csv.gz"
  run header-only fail --baseline "$work/header_only.csv.gz"

  # Not valid gzip at all: exercises the pclose status path specifically,
  # which used to be discarded outright.
  printf 'this is not gzip data' > "$work/corrupt.csv.gz"
  run corrupt-gz fail --baseline "$work/corrupt.csv.gz"

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
  run digits-inflated fail --baseline "$work/digits_up.csv.gz"

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
  run ulps-inflated pass --baseline "$work/ulps_inflated.csv.gz"
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
