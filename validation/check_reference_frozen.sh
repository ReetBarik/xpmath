#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ctest sweep_reference_frozen: the Reference baselines (xpmath 4d597fd) must
# never change. The sums are checked twice: against the SHA256SUMS file next to
# the data and against the copy hard-coded below, so changing the Reference means
# editing two places, in a commit nobody can mistake for a re-baseline.
#   usage: check_reference_frozen.sh <validation/sweep/reference_4d597fd>
set -u
dir=${1:?usage: check_reference_frozen.sh <validation/sweep/reference_4d597fd>}

# Hard-coded copy of reference_4d597fd/SHA256SUMS.
EXPECTED='82d28a0b2a5dee52d05a374c45898b40c4a48d2e5b5ece8706f58a349d3d3ad0  sweep_baseline.csv.gz
d90f1a9d250dc78e561c50aea29b0d6a0163aad4a770644637a688d8f518565c  sweep_baseline_a100.csv.gz
ef17c92397a5aad08a0d7f537d8a8744006a4747ba38645d97fa623e00283cf5  sweep_baseline_b200.csv.gz
5c37278ac0e9ef4e5e81fc7b7de32366dfac076827ea4ae8781a81343c1404d6  sweep_baseline_mi250.csv.gz
3eb691db6ded9751c9570ab734eadb788261562840cff3eadaad9fb0c6815618  sweep_baseline_mi300.csv.gz'

rc=0
cd "$dir" || { echo "FAIL: cannot cd to $dir"; exit 1; }
if [ "$(cat SHA256SUMS)" != "$EXPECTED" ]; then
  echo "FAIL: SHA256SUMS differs from the sums hard-coded in $0"; rc=1
fi
printf '%s\n' "$EXPECTED" | sha256sum -c - || rc=1
n=$(ls | grep -c '\.csv\.gz$')
if [ "$n" != 5 ]; then echo "FAIL: expected 5 .csv.gz files in $dir, found $n"; rc=1; fi
[ $rc = 0 ] && echo "reference_4d597fd: 5 files match the frozen sums"
exit $rc
