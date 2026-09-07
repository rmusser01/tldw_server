# Expat dual-copy qualification (candidate only)

Tracking: TASK-13013.7.9. Execution plan:
`IMPLEMENTATION_PLAN_task_13013_7_9_expat_dual_copy.md`.

This directory does **not** supply a production replacement or a qualified image.
Source preparation, automated authentication and the native system-build harness
are implemented; successful native builds, parser/ABI tests, combined
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

### Automated authentication gate

On a Linux verifier with `/usr/bin/gpg`, run the separate authentication command
before extraction or source execution:

```sh
python Helper_Scripts/Supply_Chain/expat_candidate.py authenticate-sources /sources \
  --public-keys /public-keys --evidence /evidence/new-authentication-run
```

`/sources` must contain the five pinned archives plus
`expat-2.8.4.tar.gz.asc` and `Python-3.12.14.tar.xz.asc`. `/public-keys` must contain
`expat-key.asc`, `python-key.asc` and `debian-maintainer-full-key.asc` (the complete
public certificates corresponding to the fingerprints above). Key-file hashes
are recorded, but do not replace checking the verified signer's identity.

Use read-only input mounts, no network, an unprivileged user, and a new writable
evidence directory. The helper creates its own temporary private keyring and
disables user configuration, automatic key retrieval/import and agent startup.
It does not import into the operator's keyring, extract source or run build code.
The eventual build controller must preserve input immutability through use;
this standalone gate is not yet connected to a native build workflow.

All five source hashes must match before GnuPG starts. Each verification must
exit successfully and emit exactly one valid signature matching **both** approved
fingerprints and the expected binary/text signature class. Expired/revoked keys,
bad or additional signatures, missing output and process errors fail closed.
`TRUST_UNDEFINED` is allowed because identity is anchored by the independently
approved fingerprints, not a local web of trust. This follows the separate
cryptographic-validity and key-status fields in
[GnuPG's status protocol](https://github.com/gpg/gnupg/blob/master/doc/DETAILS).
Offline certificates cannot prove that no newer revocation has been published.

Fresh machine status, human diagnostics, command arguments, process outcomes and
input identities are retained. A success record is written only after all three
signatures pass; an existing evidence directory is rejected to prevent stale
success records from being reused. Timeout and executable failures retain their
available diagnostics. No archived status log is accepted as authentication.

The automated helper was exercised against the real pinned inputs in the same
networkless verifier image above; all three signatures passed. Substituting the
Python signature for the Expat archive, or removing the Expat public certificate
by substituting the Python certificate, each failed without a success record.
Evidence: `/private/tmp/task-13013-7-expat-auth-real.jMY6Yy`. These bounded checks
still establish **source authentication only**, not Expat remediation.

## Native system-library qualification

The `Expat Native System Candidate Qualification` workflow uses native amd64
`ubuntu-24.04`, pinned checkout/upload actions, read-only repository permissions,
the approved Python/Trixie base digest and signed Debian snapshot. It publishes
no image and changes no production recipe or scan policy. The native controller
rejects other host/daemon/image architectures and reused evidence directories.

The image build installs build dependencies and downloads the pinned sources
and public certificates; it does not execute downloaded source. Separate
networkless containers then authenticate and prepare sources, build/test, run
ASan/UBSan tests, and check installation. Build and test containers run as UID
1000 without capabilities; only the fresh package-install container uses root
with ordinary container capabilities. None uses privileged mode or host mounts.
The controller records immutable input-image IDs and the checkout SHA, retains
phase exit codes and uploads evidence even when a phase fails.

System qualification includes:

- A full Debian 2.8.4 source-package rebuild with local candidate revision
  `2.8.4-1~deb13u1+tldw1`, rather than unstable binary packages.
- Explicit ordinary `make check`, plus verbose execution evidence for
  `test_default_attr_index_after_dtd_copy`.
- SONAME and all defined public-export comparisons for both `libexpat` and
  `libexpatw`, including unversioned symbols, and dynamic dependency checks.
- Dedicated unsigned-short wide-character controls for namespaces, defaults,
  first-declaration precedence, CDATA/NMTOKENS normalization and external-child
  parser lifecycle, with whole-buffer and small-chunk input. Upstream's test
  suite explicitly rejects Debian's unsigned-short mode; no wide upstream-suite
  pass is claimed. The external-child test retains the non-null context needed
  to exercise `dtdCopy`.
- CPU-bounded attribute scaling at 4,000/8,000/16,000 NMTOKENS attributes with
  unnormalized values. Five-sample median process CPU times are measured for
  whole-buffer and 64 KiB incremental input. The baseline must reproduce greater
  than 3× growth for the last doubling; the candidate must grow less than 3× and
  consume less than 60% of the baseline's largest-input time. Failure is retained,
  not treated as a timing exemption. CPU and address-space limits are enforced.
- Upstream ASan/UBSan tests in a separate unprivileged, networkless container.
- Fresh-container package installation, exact installed parser version checks,
  repeated wide controls, `apt-get check` and empty `dpkg --audit`.

The workflow's result is **system-only**. It records the still-unmodified Python
bundled parser and cannot admit a dual-copy release. CPython refresh/rebuild,
source SBOM regeneration, Python XML tests and combined application/rendering
qualification remain required by the implementation plan.

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

## Native bundled Python qualification

The workflow's separate Python job uses the Dockerfile's `python` target; the
system job explicitly selects `system`. Both remain candidate-only and retain
networkless, resource-bounded execution after authenticated downloads.

The Python job performs three stages:

1. Authenticate the exact inputs, extract a fresh Python 3.12.14 tree, run the
   unchanged upstream refresh procedure (only its three release assignments
   differ), and regenerate its source SBOM. The offline download adapter accepts
   only the exact approved release URL. Independently verify both preserved
   headers, the namespace include, package identities, file hashes and source
   relationships. Preserve a pre-build source archive.
2. Rebuild the full interpreter with the official image's PGO/LTO/shared-library
   flags. Run `test_pyexpat`, `test_xml_etree`, `test_xml_etree_c`, `test_minidom`
   and `test_sax` against both baseline and candidate, rejecting lost tests or new
   skips. Require explicit ElementTree import, actual loaded-runtime paths,
   non-null-context child-parser controls, bounded CPU-scaling improvement and
   ELF checks excluding system Expat linkage or unnamespaced dynamic exports.
3. Install the complete staged interpreter into a fresh container, verify its
   exact executable/library/extension hashes and repeat installed-runtime
   controls without a build-tree library path. Dependency/audit checks remain
   mandatory. No image is published or promoted.

Artifacts are `expat-python-candidate-<run-id>`. A `python-qualified.txt` marker
qualifies only that bundled parser build. System and Python qualification still
do not establish combined application compatibility, source-aware scanner
admission, or production readiness.

The successful system-only run is [34157012667](https://github.com/rmusser01/tldw_server/actions/runs/34157012667),
at commit `82c78368d4`. Python native qualification is still pending.
