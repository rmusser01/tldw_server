# Expat dual-copy qualification (candidate only)

Tracking: TASK-13013.7.9. Execution plan:
`IMPLEMENTATION_PLAN_task_13013_7_9_expat_dual_copy.md`.

This directory does **not** supply a production replacement or a qualified image.
Source preparation is implemented; native builds, parser/ABI tests, combined
application compatibility, source-aware scans and final security review remain
required. No vulnerability waiver follows from this evidence.

## Why both copies matter

The saved FFmpeg candidate manifest
`94b8ed9fa76fdc3959c7b58a5b2417277bb2171d2d29803f33ac6e593d1c7942`
contains system `libexpat1` 2.8.3-1~deb13u1 and CPython 3.12.14's separate bundled
Expat 2.8.3. The system package supports fontconfig/graphics consumers; Python XML
ingestion reaches the bundled copy. A system-only update is incomplete.

Expat 2.8.4 includes the attribute-index fix and its necessary `dtdCopy` follow-up.
Taking only the first two CVE-2026-66046 commits can introduce CVE-2026-76641.
See the [Debian tracker](https://security-tracker.debian.org/tracker/CVE-2026-66046)
and [complete upstream release](https://github.com/libexpat/libexpat/releases/tag/R_2_8_4).

## Prepared source identities

Download the five versioned archives listed in the implementation plan into a
task-owned directory. `Helper_Scripts/Supply_Chain/expat_candidate.py` verifies
their exact SHA-256 values before extraction. It does not download, extract or
execute them:

```sh
python Helper_Scripts/Supply_Chain/expat_candidate.py verify-sources /absolute/task/source-directory
```

Signature verification must precede executing any downloaded source. The three
signatures were locally verified on 2026-09-07 in an existing task-owned image,
with no network, no capabilities, read-only source inputs and a temporary keyring:

| Signed input | Signing fingerprint | Primary fingerprint |
| --- | --- | --- |
| Expat 2.8.4 upstream archive | `CB8DE70A90CFBF6C3BF5CC5696262ACFFBD3AEC6` | `3176EF7DB2367F1FCA4F306B1F9B0E909AF37285` |
| Python 3.12.14 archive | `7169605F62C751356D054A26A821E680E5FA6305` | same |
| Debian Expat 2.8.4-1 `.dsc` | `7D887DC8BA7BBBA7B835E3BADCE310E7864CC8BF` | `A0DF7E0D3851E0EE45C00BC8ACE1F33CB933BBBB` |

All three produced `VALIDSIG`; key-owner trust was not asserted. Debian's signed
`.dsc` binds the separate Debian orig/archive and packaging hashes. The upstream
release archive and Debian orig archive intentionally differ: the latter is the
upstream repository archive, not the generated release tarball.

Local evidence is in
`/private/tmp/task-13013-7-expat-sources.uuqxOa/source-signature-verification.log`.
Verifier image: `sha256:9803426ce3cc2b0b9938db476ed1b296088ad85fed8419b02dcf3aef5a94186d`.
This was a lightweight emulated signature check, **not native binary qualification**.

## CPython preparation contract

After verifying/extracting the exact Python archive into a fresh task-owned tree:

```sh
python Helper_Scripts/Supply_Chain/expat_candidate.py update-python-metadata /absolute/task/Python-3.12.14
```

This changes only the expected release/tag/hash assignments in CPython's existing
`Modules/expat/refresh.sh` and the matching Expat package fields in
`Misc/sbom.spdx.json`. It rejects unexpected baseline metadata before writing and
leaves other package identities alone. It does **not** update parser source files
or regenerate SBOM file checksums.

The native qualification must subsequently execute CPython's own refresh script,
regenerate its source SBOM, and rebuild/test Python. Retain `expat_config.h` and
`pyexpatns.h`; do not switch linking modes or overlay only a replacement pyexpat
binary. Do not use intermediate metadata as evidence of a remediated parser.

Debian's packaging rules do not explicitly invoke the parser test suite. The
native qualification must separately run upstream tests, including the
`test_default_attr_index_after_dtd_copy` regression, and preserve their actual
exit statuses. A successful package build alone is insufficient.
