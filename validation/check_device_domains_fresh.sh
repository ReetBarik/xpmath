#!/usr/bin/env bash
# Staleness guard for docs/DEVICE_PRECISION.md.
#
# Regenerates the document from the committed sweep baselines and fails if the
# result differs from what is checked in. Mirror of check_domains_fresh.sh for
# the host-only DOMAINS.md; same contract, different flag on the generator.
#
# Needs only Python 3 and the committed CSVs — no build, no Kokkos, no GPU.
#
#   validation/check_device_domains_fresh.sh   # exit 0 if fresh, 1 if stale
#
# CORE_PLAN C9 degradation rule: an arch may be missing, but the generated
# document must still mention it (as a NO BASELINE gap). The positive greps
# below refuse to pass on a document that quietly dropped a100 or mi250.
set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
doc="$root/docs/DEVICE_PRECISION.md"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if ! python3 "$root/scripts/gen_domains.py" --device-precision > "$tmp"; then
  echo "check_device_domains_fresh: generator failed" >&2
  exit 2
fi

if ! [ -s "$tmp" ]; then
  echo "check_device_domains_fresh: generator wrote nothing" >&2
  exit 2
fi

if ! [ -f "$doc" ]; then
  echo "check_device_domains_fresh: $doc does not exist" >&2
  exit 1
fi

# Every arch must be accounted for — present with data, or present with a
# NO BASELINE verdict. A zero means an arch was dropped rather than reported.
for arch in a100 mi250; do
  n=$(grep -c "$arch" "$tmp" || true)
  echo "check_device_domains_fresh: '$arch' mentions in generated doc: $n"
  if [ "$n" -eq 0 ]; then
    echo "check_device_domains_fresh: FAIL — generated doc never mentions $arch" >&2
    exit 1
  fi
done

if diff -u "$doc" "$tmp" > /dev/null; then
  echo "check_device_domains_fresh: PASS -- docs/DEVICE_PRECISION.md matches the sweep data"
  exit 0
fi

echo "check_device_domains_fresh: FAIL -- docs/DEVICE_PRECISION.md is stale." >&2
echo "  Regenerate it with:" >&2
echo "    scripts/gen_domains.py --device-precision > docs/DEVICE_PRECISION.md" >&2
echo >&2
diff -u "$doc" "$tmp" | head -60 >&2
exit 1
