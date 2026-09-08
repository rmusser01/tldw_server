# Candidate FFmpeg wheel source preparation

TASK-13013.7.12. Source preparation is **incomplete**. There is no build-ready
source lock, replacement wheel, derivative image or vulnerability clearance.
Production recipes, dependency locks, scanner policy and the existing standalone
FFmpeg 9 candidate recipe remain unchanged.

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
