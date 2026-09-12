#!/usr/bin/env bash
# Staleness guard for docs/DOMAINS.md.
#
# Regenerates the document from the committed sweep CSVs and fails if the result
# differs from what is checked in. That is the whole contract: the markdown must
# be exactly what scripts/gen_domains.py currently emits, so it cannot drift the
# way CLAUDE.md twice did.
#
# Needs only Python 3, the gzipped baseline, the grid and the open-defect
# register -- no build, no Kokkos, no libquadmath.
# Runs in well under a second, which is why it is safe to put in front of CI.
#
#   validation/check_domains_fresh.sh        # exit 0 if fresh, 1 if stale
#
# NOTE ON THE FAILURE PATH. The remediation text below is printed only when the
# doc is stale, and nothing in the suite makes it stale on purpose, so it is
# never executed by any test. It rotted once already: it used to name a bare
# `scripts/sweep_accuracy` (no --out; that spelling used to overwrite the
# committed baseline) and `--classify`, a flag that does not exist --
# `sweep_accuracy --classify` exits with "Unknown argument". Both corrected.
# If you change how the baseline is regenerated, this text does not fail when
# it goes wrong; it has to be read.
set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
doc="$root/docs/DOMAINS.md"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if ! python3 "$root/scripts/gen_domains.py" > "$tmp"; then
  echo "check_domains_fresh: generator failed" >&2
  exit 2
fi

if ! [ -f "$doc" ]; then
  echo "check_domains_fresh: $doc does not exist" >&2
  exit 1
fi

if diff -u "$doc" "$tmp" > /dev/null; then
  echo "check_domains_fresh: PASS -- docs/DOMAINS.md matches the sweep data"
  exit 0
fi

echo "check_domains_fresh: FAIL -- docs/DOMAINS.md is stale." >&2
echo "  Regenerate it with:  scripts/gen_domains.py > docs/DOMAINS.md" >&2
echo "  If a numeric fix landed, re-baseline the sweep FIRST (own commit):" >&2
echo "    <build>/tests/sweep_accuracy --out validation/sweep/sweep_baseline.csv" >&2
echo "    gzip -9 -f validation/sweep/sweep_baseline.csv" >&2
echo >&2
diff -u "$doc" "$tmp" | head -60 >&2
exit 1
