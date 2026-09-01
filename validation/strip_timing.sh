#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-DHB-License
# SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
#
# Byte-identical gate — timing stripper (layout-independent).
#
# WHY THIS REPLACES validation/s3/strip_timing.sh
# -----------------------------------------------
# The S2/S3 stripper hard-coded the DD table shape: `NF == 10` pipe fields with
# accuracy in fields 6-9. That holds for kokkos_ep_demo and kokkos_ep_demo_complex
# and fails for every other demo. The FF tables place two backends side by side and
# carry 18 fields, so the `NF == 10` rule never matched, every row passed through
# verbatim, and the timing columns survived into the "stripped" output. Diffing two
# such captures then reports dozens of differences that are nothing but run-to-run
# wall-clock jitter — the gate reports a failure it should not, and could hide a real
# accuracy change inside the noise it generates.
#
# Discovered during S5 phase 1 (FF): that phase's diffs were unusable until the
# accuracy columns were extracted by hand.
#
# HOW THIS ONE WORKS
# ------------------
# It keys on NUMBER FORMAT rather than column position. Both timing spellings used in
# this repo are distinguishable from accuracy at a glance:
#
#     DD demos     timing is a slowdown ratio with an x suffix   4.6x   15.6x
#     FF/QF demos  timing is a nanosecond figure, four decimals  4.4139  1158.5259
#     ALL demos    accuracy is digits, exactly two decimals      13.96   31.00
#
# So on a data row: drop any field matching the two timing spellings, keep the rest.
# Non-data rows (banners, rules, headers) pass through unchanged, so a structural
# change — an op appearing, disappearing, or being reordered — still shows in the
# diff rather than being silently normalized away.
#
# A row is only treated as data if at least one accuracy field survives; otherwise it
# prints verbatim. That makes the script fail loudly rather than silently emptying a
# table it does not understand.
#
# The banner line carrying `batch=... timing=...` is dropped: it names the timing mode
# and is not accuracy content.
#
# Verified against every capture pair committed in validation/ (S2 and S3 DD real and
# complex, S5 FF real and complex).
#
# Usage: strip_timing.sh <demo-output.txt>

set -euo pipefail

awk -F'|' '
  # Drop the banner line that names the timing mode.
  /batch=.*timing=/ { next }

  {
    # A data row has pipes and a leading " opname " field.
    if (NF > 2 && $1 ~ /^ *[A-Za-z][A-Za-z0-9_ ()]*$/) {
      out  = $1
      kept = 0
      for (i = 2; i <= NF; i++) {
        f = $i
        gsub(/^[ \t]+|[ \t]+$/, "", f)              # trim

        if (f ~ /^-?[0-9]+(\.[0-9]+)?x$/)  continue # DD timing: slowdown ratio "4.6x"
        if (f ~ /^-?[0-9]+\.[0-9]{4}$/)    continue # FF/QF timing: ns, four decimals
        if (f == "" && i == NF)            continue # trailing empty field

        out = out "|" $i
        if (f ~ /^-?[0-9]+\.[0-9]{2}$/) kept++      # accuracy: exactly two decimals
      }
      # Only accept it as a data row if accuracy fields actually survived.
      if (kept > 0) { print out "|"; next }
    }
    print
  }
' "$1"
