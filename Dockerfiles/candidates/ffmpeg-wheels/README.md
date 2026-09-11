# Candidate FFmpeg wheel source preparation

TASK-13013.7.12. Source preparation is **incomplete**. There is no build-ready
source lock, replacement wheel, derivative image or vulnerability clearance.
Production recipes, dependency locks, scanner policy and the existing standalone
FFmpeg 9 candidate recipe remain unchanged.

## Deterministic source-input gate

[source-inputs.py](source-inputs.py) implements the candidate-only
**ffmpeg-wheel-source/v1** metadata and byte-binding gate. It is a pure local
validator: it does not fetch sources, invoke signature tools, apply patches,
build wheels or execute any input. A trusted acquisition step must provide both
the source root and the normalized original-match records. The production
adapter that derives the complete 43-record **{input_id, cve, owner}** set
remains pending; the validator neither reads nor interprets retained scanner
output.

Version 1 uses these exact object shapes. Unknown or missing fields fail closed:

- The lock has **schema_version**, **inputs**, **patches**, **coverage**,
  **builds** and **evidence**.
- An input has **name**, HTTPS **url**, **sha256**, **path** and
  **authentication**. Authentication has exactly **method** and a non-empty
  **evidence_paths** list.
- An ordered patch has full **commit**, **sha256**, **path**, **requires** and
  **source_paths**. Every prerequisite must occur earlier in the patch array.
- A coverage record has **input_id**, **cve**, **owner**, **source_paths**,
  **disposition**, **repair_commits**, **evidence_paths** and
  **regression_id**. Disposition is one of **repaired**, **already_fixed** or
  **absent_condition**.
- A build has **owner**, **version**, **abi**, **platform**, **tools**,
  **assets** and **configuration**. Platform is exactly native
  Linux/amd64/Python 3.12. Configuration has a unique **environment** and
  **evidence_paths**.
- An evidence record has only **path** and **sha256**. Original-match records
  have only **input_id**, **cve** and **owner**.

All identifiers and paths are unique where they declare identities. Paths are
canonical relative POSIX paths; absolute paths, traversal, backslashes, symlinks
in the root or a declared path, and non-regular files are rejected. The complete
set of regular files below the root must equal the input, patch and evidence
paths in the lock, so an extra unreviewed file also fails. Every evidence file
is hash-bound and referenced by authentication, coverage or build
configuration. Every input is referenced by one of the two build records.

Coverage must equal the normalized original-match set by the full
**{input_id, cve, owner}** identity, not by input ID alone. A **repaired**
record must list known commits in declared order, include their prerequisites
and bind its source paths to those patches. The other dispositions must have no
repair commits. All dispositions require hashed evidence and a unique
regression ID. Build records are fixed to **av 18.1.0 / cp311-abi3** and
**opencv-python 5.0.0.93 / cp37-abi3**, with distinct environments and declared
input, configuration and asset references.

An authentication record only binds a descriptive method to hashed evidence.
The parser does **not** establish cryptographic authenticity, validate a signer
or promote signer text into trust. Signature/key/registry verification belongs
to trusted acquisition, and its machine-readable result must be retained as a
hashed evidence file.

On success, **verify_sources()** returns the schema version, sorted coverage
IDs, the declared input hashes, and **source_lock_sha256**. The last value is
the SHA-256 of UTF-8 JSON serialized with sorted object keys, compact separators
and arrays in their declared order. It is the semantic identity of the v1 lock
(whitespace and object-key order do not affect it); it is separate from every
raw artifact SHA-256. Array order is part of the v1 lock semantics.

The CLI prints JSON to stdout only after validation succeeds:

    python Dockerfiles/candidates/ffmpeg-wheels/source-inputs.py \
      --lock /path/to/source-lock.json \
      --root /path/to/source-root \
      --original-matches /path/to/normalized-original-matches.json

Malformed or incomplete input exits nonzero without a success document. No real
**source-lock.json** is supplied by this checkpoint. Actual upstream patch
application, regression execution, complete source-ledger review and owning
wheel builds remain separate qualification work.

The approved design is
[`Docs/Design/2026-09-07-task-13013-7-12-ffmpeg-wheel-remediation.md`](../../../Docs/Design/2026-09-07-task-13013-7-12-ffmpeg-wheel-remediation.md).
It preserves PyAV 18.1.0 and OpenCV 5.0.0.93 while rebuilding their private
FFmpeg libraries from a complete repaired 8.x source baseline. The implementation
plan and Backlog task track qualification separately from source authentication.

## Authenticated baseline

The [official release page](https://ffmpeg.org/download.html) identifies FFmpeg
8.1.2 as an 8.1 release and publishes release-signing fingerprint
`FCF986EA15E6E293A5644F10B4322F04D67658D8`.
The following exact bytes were acquired and verified during source preparation:

| Input | SHA-256 |
| --- | --- |
| [Source archive](https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz) | `464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c` |
| [Detached signature](https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz.asc) | `0a0963fccd70597838073f3e31b20f4a4d8cc2b5e577472c9a5a1f22624246f8` |
| [Public signing key](https://ffmpeg.org/ffmpeg-devel.asc) | `397b3becedcd5a98769967ff1ff8501ddc89f8368b8f766e4701377d7dbaabe5` |

GPG returned exit 0 and this machine-readable signature status:

```text
[GNUPG:] VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8 2026-06-17 1781664539 0 4 0 1 10 00 FCF986EA15E6E293A5644F10B4322F04D67658D8
```

The verification used an isolated temporary keyring in the existing task-owned
image `sha256:9803426ce3cc2b0b9938db476ed1b296088ad85fed8419b02dcf3aef5a94186d`,
without network, as UID/GID 65534, with read-only root and input mount, dropped
capabilities and bounded scratch/resources. The keyring's `TRUST_UNDEFINED`
warning is retained: authenticity was checked against the independently read
official fingerprint, not an established local web of trust. This was an emulated
amd64 signature check, not native compilation or runtime qualification.

With these acquired files in a read-only input directory and a fresh writable
keyring, the verification commands are:

```sh
gpg --homedir /tmp/gnupg --batch --import /input/ffmpeg-devel.asc
gpg --homedir /tmp/gnupg --batch --with-colons --fingerprint FCF986EA15E6E293A5644F10B4322F04D67658D8
gpg --homedir /tmp/gnupg --batch --status-fd 1 --verify /input/ffmpeg-8.1.2.tar.xz.asc /input/ffmpeg-8.1.2.tar.xz
```

These diagnostic commands require checking the expected hashes, exact fingerprint
and `VALIDSIG` result; they are not a substitute for the planned fail-closed input
validator. Local acquired bytes currently reside at
`/private/tmp/task-13013-7-12-source.MJwOvz`. A durable hash-bound evidence artifact
is required before CI may depend on those bytes.

## Repair coverage checkpoint

Static forward-apply checks against the authenticated, unmodified 8.1.2 release
accept the previously retrieved upstream repairs for these 19 CVEs:
CVE-2026-64830 through -64835; -65703, -65704, -65706; -66037 through -66041;
and -70628 through -70632. For RSCC and CineForm the checks use the actual repair
commits, not their feature-introduction references. Equivalent upstream/backport
patches were checked as alternatives, not as a series to apply twice.

No patch was applied by these checks. A successful `git apply --check` establishes
context compatibility only; it does not establish complete repair, prerequisites,
ABI compatibility, exploitability or regression success. The source ledger must
still reconcile every original wheel/component input independently.

The independent source investigator has identified floodfill's complete repair
`24c322fdb232d0a3f3790d544dcb64e5c2138e79`, hqdn3d's prerequisite
`f0f634b6585fdc7bbb43ab3ae461499bfca9ad2e` followed by dynamic-size support
`5d7112c60e6f0f0742ce47d448e6da0718a70f4c`, and the three MagicYUV repairs
already present in 8.1.2. Parent reconciliation of that sequence and regression
coverage remains required. No `source-lock.json` will be labeled ready while
source, build or qualification inputs remain unresolved.

The security investigator's final handoff encountered a platform safety
restriction. The above interim findings are not a completed reviewed patch
package. That security path is paused; no alternate route or retry was used to
bypass the restriction, and no repair series has been applied.

The original Grype report SHA-256 was reverified as
`2bba9bbad78ea5f0cb3dc2c8d3e645185a83f8c4a18665e1bb323965913dd194`.
All 43 unique triage input identities remain present and `needs_review` in the
retained diagnostics at `/private/tmp/task-13013-7-combined-scans.vpkHOr`.
No scanner result or applicability status has been suppressed or changed.

## Owning-package source and build boundaries

The exact official source distributions are available. Downloaded bytes were
independently checked against the retrieved PyPI version metadata:

| Owner | Source archive SHA-256 |
| --- | --- |
| [PyAV 18.1.0](https://pypi.org/pypi/av/18.1.0/json) | `47bfc286e1bc9de7ab4681fc2b575cd2460a66919d31ffe1bd5aa54fae531a28` |
| [OpenCV 5.0.0.93](https://pypi.org/pypi/opencv-python/5.0.0.93/json) | `66aac3e5b5faa48d4025816592f3af19e4bfc2c68dec067bae2dbb4ca10aa9e2` |

Registry metadata/hash verification is not a release-signature claim. Sources and
reference wheels are retained locally at
`/private/tmp/task-13013-7-12-packaging.1Oyq28`; durable input retention remains
required. No source-backed version or feature impossibility was found in the
bounded packaging inspection, but native build compatibility is unproven.

PyAV's Python build requires setuptools >=77; OpenCV's Python 3.12 build requires
setuptools <70. They need separate build environments, not a shared resolver or
runtime dependency change. PyAV/OpenCV also use different native closures (including
different VPX ABIs and TLS providers); do not share one system FFmpeg library.
Compiler images, build tools and optional OpenCV assets still need immutable
pins and source/license evidence. Headless, disabled IPP/Orbbec, or omitted Qt
plugins/fonts are not acceptable shortcuts for the original desktop wheel.

The inspected [PyAV vendor patch](https://github.com/PyAV-Org/pyav-ffmpeg/blob/a71bf9279f7a4659154b68ba6783e89be460bcd5/patches/ffmpeg.patch)
moves x264/x265 from FFmpeg's GPL dependency list into its version3 list. The
retained binary enables those codecs but reports LGPLv3. Do not copy this
license-category rewrite into the candidate. The upstream-supported configuration
to evaluate is `--enable-gpl --enable-version3`, retaining the codecs and truthful
third-party notices and corresponding source records. Compare capabilities
explicitly because GPL configuration can enable additional components; this is
not legal clearance or permission to change repository licenses. The current
root `LICENSE` and `pyproject.toml` identify server code as GPL-3.0-only; older
overview text is not the authoritative license map.

## Compiler-image metadata checkpoint

Registry manifest bytes, registry digest headers, configuration hashes and declared
platforms were independently compared for these candidate build inputs:

| Role | Manifest reference | Configuration SHA-256 |
| --- | --- | --- |
| OpenCV dated recipe tag `20260531` | `quay.io/opencv-ci/opencv-python-manylinux_2_28-x86-64@sha256:26c8159d923fbaa3c893aa53cdf67d5b2da7f858daf388a494a282ca118b83c9` | `ae0e52e7f6964b35505af87ff3329ac86ddd6a56bd4f47580107d6b978588e37` |
| cibuildwheel 4.2.0 native x86_64 default, candidate for PyAV | `quay.io/pypa/manylinux_2_28_x86_64@sha256:012f4a50472412f18bb2b450c1cce7158434cfae4ae878591c2748a13a30c2be` | `e0669ff0af3896498da5f503bfa032499f1ddb127109956b95db45856a28b7a1` |

Both are single-image schema-2 manifests declaring `linux/amd64`, not indexes.
Only metadata was fetched; layers, compiler execution and package inventories
were not verified. Digest identity is not a signature or build attestation.
OpenCV's [packaging workflow](https://github.com/opencv/opencv-python/blob/b83046cda41133f1bf2e73e99dba16a1248f103a/.github/workflows/build_wheels_manylinux.yml)
selects the dated tag; the digest records its observed resolution, not proof the
tag has never moved. The PyAV candidate comes from cibuildwheel's
[pinned image map](https://github.com/pypa/cibuildwheel/blob/1828c10ab37f080699c7b81cea34097c684a7074/cibuildwheel/resources/pinned_docker_images.cfg).
PyAV's original release workflow did not pin cibuildwheel, so original-run image
attribution remains unresolved. Do not label this candidate as that proven image
or substitute today's `latest` tag. Metadata is retained beside the owning-source
archives in the packaging scratch directory above.

## Offline OpenCV asset checkpoint

These checks use the exact OpenCV source distribution, not a changed build
configuration. They do not turn off supported capabilities.

- **IPPICV:** the Linux x86_64 branch of `opencv/3rdparty/ippicv/ippicv.cmake`
  selects [this archive](https://raw.githubusercontent.com/opencv/opencv_3rdparty/406d398c436d0465c8e53dd432d9ecd9301d5f4a/ippicv/ippicv_2026.0.0_lnx_intel64_20260327_general.tgz).
  Downloaded SHA-256 is `5198d6f76e61d5aa0fa5dde72cc22947af1a86efbf49d08b90d374699c811fd0`;
  its MD5 matches upstream `9a3ee0c5c3c02102faa422d60bfd1f4a`. The latter is a
  compatibility check for OpenCV's existing cache, not the new cryptographic
  identity. The archive includes `ippicv_lnx/EULA.txt`, whose SHA-256 is
  `c1bf7165226036081d5cbe4baca565eb5e817fe56b9cf1ac7247688feb32b3e4`.
  Preserve that notice; hashing it is not legal clearance. With
  `OPENCV_DOWNLOAD_PATH` pointing to an isolated pre-populated cache, the existing
  cache key is `ippicv/9a3ee0c5c3c02102faa422d60bfd1f4a-ippicv_2026.0.0_lnx_intel64_20260327_general.tgz`.
  A future controller must verify SHA-256 before placing bytes in that cache and
  enforce offline configuration/build; this checkpoint does not implement it.
- **ITT:** `cmake/OpenCVDetectTrace.cmake` adds the source distribution's
  `3rdparty/ittnotify` directory when `WITH_ITT` and `BUILD_ITT` are enabled.
  Its source/header/license files are already covered by the verified sdist.
  No separate ITT download is required by that branch.
- **Orbbec:** the original wheel's embedded build information says `Orbbec: YES`,
  not `Orbbec SDK: YES`. The source emits distinct labels for those branches.
  `WITH_OBSENSOR` defaults on while `OBSENSOR_USE_ORBBEC_SDK` defaults off; the
  Linux non-SDK branch checks `linux/videodev2.h` and uses V4L2. Preserve that
  branch and its native compatibility checks. The separately selectable SDK
  download is not required merely to reproduce this observed baseline.
- **libyuv:** the compiler-image recipe builds libavif 1.4.2 with
  `AVIF_LIBYUV=LOCAL`. The release tag resolves to commit
  `c5240fc79fe5c2407e10afd35f5505ef6333ea49`; its
  [LocalLibyuv.cmake](https://github.com/AOMediaCodec/libavif/blob/c5240fc79fe5c2407e10afd35f5505ef6333ea49/cmake/Modules/LocalLibyuv.cmake)
  pins `644251f252a84bf8ce91ff0aca86a9b16b069ab8`. The downloaded
  [exact source archive](https://chromium.googlesource.com/libyuv/libyuv/+archive/644251f252a84bf8ce91ff0aca86a9b16b069ab8.tar.gz)
  has SHA-256 `037ac232b25d7cc56dee6a948ddfbb4450465395b029debea65e4b88a214881e`
  and declares `LIBYUV_VERSION 1924`. This is source-input evidence for the image's
  dependency, not proof of the bytes compiled into its uninspected layers. The
  existing libavif code accepts pre-acquired source at `ext/libyuv`; its fallback
  network fetch must not occur during a future offline build.

Downloaded asset bytes and inspected libavif metadata are retained at
`/private/tmp/task-13013-7-12-assets.DJcuFZ`. Full build closure, durable artifact
retention, native execution and the separately restricted security review remain
incomplete. No image layers were pulled and no asset or package code was executed
by this metadata checkpoint.
