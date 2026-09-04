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
echo "  If a numeric fix landed, re-baseline the sweep FIRST:" >&2
echo "    scripts/sweep_accuracy" >&2
echo "    scripts/sweep_accuracy --classify validation/sweep/sweep_classified.csv" >&2
echo >&2
diff -u "$doc" "$tmp" | head -60 >&2
exit 1
