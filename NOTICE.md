# NOTICE

This repository combines Reet-authored Kokkos code (Apache-2.0) with C++/Kokkos
ports of two upstream extended-precision packages: DDFUN, authored by David H.
Bailey (DHB-License, a modified-BSD-3-Clause variant), and QD, authored by Yozo
Hida, Xiaoye S. Li, and David H. Bailey at Lawrence Berkeley National Laboratory
(LBNL-BSD-License). These are **different licenses** and are tracked separately.

It is therefore **multi-licensed**:

- **Four DDFUN-derived headers** — DD and FF, real and complex — carry
  `LicenseRef-DHB-License`.
- **Two QD-derived headers** — QF, real and complex — carry
  `LicenseRef-LBNL-BSD-License`.
- **One Kokkos-extension header** carries `Apache-2.0 WITH LLVM-exception` to
  match Kokkos itself.
- **Everything else** is Apache-2.0 by default.

The mapping below is authoritative.

## Per-file license mapping

| File(s) | License | Notes |
|---|---|---|
| `third_party/include/dd_math.hpp` | DHB-License | C++/Kokkos port of DDFUN v04 (real). See `LICENSES/LicenseRef-DHB-License.txt`. |
| `third_party/include/dd_complex.hpp` | DHB-License | C++/Kokkos port of DDFUN v04 (complex). See `LICENSES/LicenseRef-DHB-License.txt`. |
| `third_party/include/ff_math.hpp` | DHB-License | DD→FF mechanical translation; PORT_NOTES.md documents FP32-specific fixes. See `LICENSES/LicenseRef-DHB-License.txt`. |
| `third_party/include/ff_complex.hpp` | DHB-License | DD→FF mechanical translation; PORT_NOTES.md documents FP32-specific fixes. See `LICENSES/LicenseRef-DHB-License.txt`. |
| `third_party/include/qf_math.hpp` | LBNL-BSD-License | C++/Kokkos port of QD 2.3.24 quad-double (real), retargeted from 4×FP64 to 4×FP32. **Not** a DDFUN derivative. See `LICENSES/LicenseRef-LBNL-BSD-License.txt`. |
| `third_party/include/qf_complex.hpp` | LBNL-BSD-License | QF complex layer composed on the QD-derived real four-word algorithms. **Not** a DDFUN derivative. See `LICENSES/LicenseRef-LBNL-BSD-License.txt`. |
| Everything else (demos, tests, harness, corpus, scripts, docs) | Apache-2.0 | Covered by the top-level `LICENSE`. |

## The DDFUN-derived files

`third_party/include/dd_math.hpp` and `third_party/include/dd_complex.hpp`
(double-double), together with `third_party/include/ff_math.hpp` and
`third_party/include/ff_complex.hpp` (float-float), are derivative works of
DDFUN v04. The FF headers are a mechanical translation of the DD headers from
2×FP64 to 2×FP32 (see `PORT_NOTES.md` for the FP32-specific fixes); as
modifications of a DDFUN derivative they inherit the DHB-License unchanged under
its §3 grant-back. All four:

- Original author: David H. Bailey (Lawrence Berkeley National Lab,
  retired / University of California, Davis).
- Original source:
  <https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz>
- Original license: **DHB-License** — a modified BSD-3-Clause variant with
  a non-standard §3 grant-back clause. Copyright (c) 2024 David H. Bailey.
  All rights reserved. Full verbatim text:
  `LICENSES/LicenseRef-DHB-License.txt` (SPDX identifier
  `LicenseRef-DHB-License`) or
  <https://www.davidhbailey.com/dhbsoftware/DHB-License.txt>.

Because DDFUN is licensed under the DHB-License, this repository cannot
relicense the ported files. The port is a derivative work and inherits
DDFUN's DHB-License terms unchanged.

### DHB-License §3 grant-back — what it means for you

DHB-License §3 states that if you publish modifications or enhancements
to the DDFUN-derived files publicly (as this repository does), without a
separate written license agreement, you thereby grant David H. Bailey a
non-exclusive, royalty-free, perpetual license to install, use, modify,
prepare derivative works, incorporate into other software, distribute,
and sublicense those enhancements, in binary and source form.

In plain terms: publishing improvements to `dd_math.hpp` /
`dd_complex.hpp` / `ff_math.hpp` / `ff_complex.hpp` publicly grants David H.
Bailey a perpetual license to incorporate those improvements back into DDFUN.
Anyone who redistributes the DDFUN-derived files from this repository inherits
this term.

### Commercial-use pointer

The DDFUN website asks commercial users to contact the author. This is
not a licensing requirement in the DHB-License text itself, but it is
worth honoring: contact David H. Bailey at <dhbailey@lbl.gov> for
commercial-use questions about DDFUN.

## The QD-derived files

`third_party/include/qf_math.hpp` and `third_party/include/qf_complex.hpp`
(quad-float, 4×FP32) are derivative works of **QD 2.3.24**, not of DDFUN. QD is
a separate package with a separate license and a different author set, so these
two files are tracked apart from the DDFUN-derived headers above. Both:

- Original authors: Yozo Hida, Xiaoye S. Li, and David H. Bailey, Lawrence
  Berkeley National Laboratory.
- Original source:
  <https://www.davidhbailey.com/dhbsoftware/qd-2.3.24.tar.gz>
- Original license: **LBNL-BSD-License**. Copyright (c) 2003-2023 The Regents of
  the University of California, through Lawrence Berkeley National Laboratory.
  Full verbatim text: `LICENSES/LicenseRef-LBNL-BSD-License.txt` (SPDX
  identifier `LicenseRef-LBNL-BSD-License`).

The port retargets QD's four-word algorithms — Priest renormalization,
Hida-Li-Bailey sloppy/ieee addition, sloppy multiplication, long division, and
Heron square root — from 4×FP64 to 4×FP32. As a derivative work it inherits the
LBNL-BSD-License; this repository cannot relicense it. FP32-specific deviations
are documented in `docs/PORT_NOTES_QF.md`.

The DHB-License §3 grant-back described above does **not** apply to these two
files: the LBNL-BSD-License contains no equivalent clause.

## Contacts

- **DDFUN questions / DHB-License / commercial use:** David H. Bailey,
  <dhbailey@lbl.gov>.
- **QD questions / LBNL-BSD-License:** the QD authors at Lawrence Berkeley
  National Laboratory; David H. Bailey, <dhbailey@lbl.gov>, is a co-author of
  both packages.
- **This repository:** contact Reet (repository owner).

## Cross-references

- Full DHB-License text: `LICENSES/LicenseRef-DHB-License.txt`.
- Full LBNL-BSD-License text: `LICENSES/LicenseRef-LBNL-BSD-License.txt`.
- Apache-2.0 text: `LICENSE` (mirrored at `LICENSES/Apache-2.0.txt`).
- Licensing rationale and Phase 2/3 open questions:
  `docs/TEST_SUITE_PLAN.md`, "Licensing" section.
