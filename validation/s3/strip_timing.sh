#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-DHB-License
# SPDX-FileCopyrightText: Copyright (c) 2026 UChicago Argonne, LLC
#
# S2 byte-identical gate — timing stripper.
#
# The demo tables interleave two things: slowdown-vs-FP64 columns, which are
# wall-clock measurements and legitimately differ run to run, and accuracy
# columns, which are pure arithmetic and must NOT differ across a mechanical
# restructure. This script deletes the former and keeps the latter, so a plain
# `diff` of two stripped captures is exactly the gate the plan specifies.
#
# Table row layout (pipe-delimited):
#   | op | Min | Max | Med | Mean |  Min  |  Max  |  Med  | Mean |
#     $1    $2    $3    $4    $5     $6     $7      $8      $9
#          <------ slowdown, dropped ---->  <---- accuracy, kept ---->
#
# Rows that are not data rows (banner, rules, headers) pass through unchanged,
# except the header/banner line carrying `batch=... timing=...`, which names
# the timing mode and is not accuracy content. Everything else is preserved so
# a structural change (an op appearing, disappearing, or being reordered) still
# shows up in the diff rather than being silently normalized away.
#
# Usage: strip_timing.sh <demo-output.txt>

set -euo pipefail

awk -F'|' '
  # A data row has 10 pipe-separated fields and a leading " opname " field.
  NF == 10 && $1 ~ /^ [A-Za-z]/ {
    printf "%s|%s|%s|%s|%s|\n", $1, $6, $7, $8, $9
    next
  }
  { print }
' "$1"
