# Expat dual-copy candidate qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for staged execution; the security-fix parent owns implementation and uses fresh read-only investigation/review.

**Goal:** Qualify Expat 2.8.4 for both the Trixie system library and CPython 3.12.14's bundled parser without adopting or publishing a production replacement.

**Architecture:** Rebuild the complete Debian 2.8.4 source package on the already-approved signed Trixie snapshot. Refresh CPython's bundled sources with CPython's own `Modules/expat/refresh.sh`, retaining its symbol namespace and configuration, then rebuild the same Python version. Test both copies and preserve source-to-binary evidence independently of scanner disposition.

**Tech Stack:** Debian source packages, Bash, CPython, Expat, pytest, Docker, native amd64 GitHub Actions.

**Spec:** User-approved candidate-only dual-copy remediation recorded in TASK-13013.7.9; independent source-to-sink investigation recorded in that task. This plan does not authorize production promotion.

## Global constraints

- Only this isolated worktree and task-owned temporary artifacts may change. Do not run heavyweight local Docker builds or alter other tasks' containers/environments.
- Pin the approved base to `python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`.
- Reuse `Dockerfiles/candidates/ffmpeg/debian.sources` at snapshot `20260906T000000Z`; retain APT signature verification. Do not add unstable binary repositories.
- No production Dockerfile, dependency lock, runtime admission, scanner thresholds, allowlists, release publishing or PR Change summary changes.
- Complete upstream 2.8.4 includes the attribute-index fix and `dtdCopy` follow-up; the first two CVE-2026-66046 commits alone are not sufficient.
- Keep CPython's `expat_config.h` and `pyexpatns.h`. Do not switch Python to system Expat or replace just a shared object without rebuilding/testing its owning interpreter.
- Use `/private/tmp/task-13013-7-pyjwt-verification-venv` before Python/pytest/Bandit; no shared environment installs.

## Stage 1: Authenticated source preparation
**Goal:** A fail-closed, reproducible source-input contract for both copies.
**Success Criteria:** Every downloaded archive matches its independently obtained pinned digest before extraction; CPython refresh and SBOM metadata agree; altered inputs fail before writes.
**Tests:** Real hash rejection tests, metadata mutation rejection, preservation of unrelated SBOM packages; verified public source archive integration.
**Status:** Complete

**Files:** Create `Helper_Scripts/Supply_Chain/expat_candidate.py`, `tldw_Server_API/tests/Supply_Chain/test_expat_candidate.py`, and `Dockerfiles/candidates/expat/README.md`.

- [x] Write tests before implementation for `verify_sources(directory: Path) -> dict[str, str]` and `update_python_metadata(root: Path) -> None`. Require missing, corrupt and symlinked archive rejection. Require an unexpected CPython baseline, duplicate Expat SBOM package, or inconsistent baseline checksum to fail before either file is changed.
- [x] Implement verification of the five exact archives below. The helper must never download, extract, install or execute downloaded code. Only `refresh.sh`'s three expected assignment values and the matching Expat SBOM package metadata may change.
- [x] Run focused pytest and Bandit. Run `verify-sources` on the real downloaded bundle at `/private/tmp/task-13013-7-expat-sources.uuqxOa`; this establishes hash identity, not binary qualification. Separately, all three source signatures passed local OpenPGP verification; README records fingerprints and evidence scope.

Checkpoint: 13 tests passed (4 existing repository warnings), Black/compileall passed, implementation Bandit 94 LOC/zero findings. Test-inclusive Bandit reported 15 intentional pytest assertions only; no scanner configuration changes. Independent source-preparation review found no actionable issues. Native execution and final security-fix review remain outstanding.

| Input | SHA-256 |
| --- | --- |
| Python-3.12.14.tar.xz | `5c8462af5790baf43a321a1559dbe0db06d1be4300fb85fb53c40060668e548a` |
| expat-2.8.4.tar.gz | `b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36` |
| expat_2.8.4-1.dsc | `b3dc30ff68a32b95746899d2c8e03cfbb5350b982916d649fab179e56ef5ed3e` |
| expat_2.8.4.orig.tar.gz | `a8a9c5cbba9110000b13cc9943f50fcd7e552a5cbad49cb191142c500a0a11b7` |
| expat_2.8.4-1.debian.tar.xz | `a90e0731e6ccdee5f4368a69ab12cf8cc9f5f29e7e61959e2e839d4ca00361fc` |

## Stage 2: Native build and parser qualification
**Goal:** Complete system and bundled fixes with executable regression/compatibility evidence.
**Success Criteria:** Native amd64 source-package build, explicit upstream parser tests, CPython rebuild/test suite, ABI and installed-copy probes pass. Missing evidence or any failure blocks qualification.
**Tests:** Debian `make -C build check` (the packaging rules do not run it), CMake sanitizer tests, CPython XML tests, bounded attribute-scaling and legitimate controls.
**Status:** In Progress

**Files:** Create `Dockerfiles/candidates/expat/Dockerfile`, `Dockerfiles/candidates/expat/qualify.sh`, `.github/workflows/expat-candidate.yml`; extend the Stage 1 test file for controller failure propagation.

Authentication checkpoint: the Stage 1 helper now offers a separate
`authenticate-sources` command. It verifies the five hashes before GnuPG, uses a
fresh private keyring, checks the exact signer/primary pairs and retains fresh
process evidence. The real signed input bundle passes; wrong-archive-signature
and wrong-public-key substitutions fail. Native workflow integration, builds,
parser tests and binary qualification below remain outstanding.

Native system harness checkpoint: `expat-candidate.yml` now orchestrates separate
offline prepare/build/sanitizer/install phases and retains each result. Wide
testing uses purpose-built unsigned-short controls: upstream `tests/common.h`
rejects Debian's `XML_UNICODE` mode, so `make -C buildw check` cannot be honestly
used as a gate. The dedicated controls cover the non-null-context `dtdCopy`
regression as well as normal XML semantics. CPython native build integration is
still outstanding; system-only evidence must never be reported as dual-copy
qualification.

First native run `34155704913` authenticated all sources, then exposed a
read-only archive collision: the redundant pre-copy occupied `dpkg-source`'s
destination. Removed only that copy; the regression failed with exit 13 before
the fix and passes afterward. Local supply-chain verification: 323 passed,
one existing opt-in skip; native re-execution remains required.

Second native run `34156111173` passed preparation, Debian source/binary builds,
all 4,884 verbose parser checks, both system ABIs, wide controls, bounded
whole/incremental scaling, and the ASan/UBSan suite. The 16,000-attribute control
took about 0.061 seconds on 2.8.3 versus 0.014 seconds on 2.8.4. All 13 retained
source/binary artifact checksums verified locally. Installation stopped because
APT's `--no-download` path rejected relative package filenames; use absolute
`/candidate/*.deb`, retaining offline installation and all evidence gates.

- [ ] Test that controller preparation/build/test/install failures propagate and evidence upload still runs; reject non-native hosts/images, missing status files and missing parser-test output.
- [ ] Use native `ubuntu-24.04`; pin checkout/upload actions to the same verified commits as util-linux qualification. Preserve read-only repository permissions, exact checkout SHA, image identities and logs. No registry push, privileged mode, Docker socket bind, host secrets or production deploy.
- [ ] Authenticate Debian `.dsc` and upstream/Python detached signatures in an isolated keyring, retaining full signer fingerprints and status output; never present GitHub's tag verification as local archive-signature verification. Abort before source execution if authentication fails.
- [ ] Extract the authenticated Debian package with `dpkg-source -x`, set an explicit local candidate package revision, and build against Trixie only. Run `DEB_BUILD_OPTIONS=parallel=4 dpkg-buildpackage -us -uc -b` followed by `make -C build check`; no `nocheck`/`stage1` profiles. Test both ordinary and wide-character builds. Retain `test_default_attr_index_after_dtd_copy` registration and execution evidence.
- [ ] Call the metadata updater, execute the exact CPython refresh script under Bash 4+, then use CPython's source SBOM generator. Verify Expat's new version/hash/CPE and per-file checksums; retain all unrelated package identities. Compare `expat_config.h` and `pyexpatns.h` to the authenticated baseline.
- [ ] Rebuild Python 3.12.14 using the official image's configure/build flags (`--enable-loadable-sqlite-extensions --enable-optimizations --enable-option-checking=fatal --enable-shared --with-lto --with-ensurepip`) and Trixie dependencies. Run `./python -m test -v test_pyexpat test_xml_etree test_xml_etree_c test_xml_dom_minidom test_sax`; preserve failure statuses and unexpected skips.
- [ ] Run bounded synthetic whole-buffer and incremental XML controls: namespaces, default attributes, first-declaration precedence, CDATA/NMTOKENS normalization and external-subparser lifecycle. Attribute-scaling input is generated locally, CPU/time bounded, and compared to the baseline; no downloaded exploit.
- [ ] Run upstream ASan/UBSan tests in a separate unprivileged, networkless build container. Compare system SONAME and all public exports (including unversioned ones), and check for missing dynamic dependencies.
- [ ] Install only into a fresh candidate container. Assert both `ctypes` system `XML_ExpatVersion()` and `pyexpat.EXPAT_VERSION` report 2.8.4; record pyexpat path, hash, configure args and DT_NEEDED. Run `apt-get check` and require empty `dpkg --audit`.

## Stage 3: Exact combined candidate evidence and review
**Goal:** Establish that the candidate preserves application behavior and that evidence covers both embedded and system sources.
**Success Criteria:** Actual combined candidate XML/rendering tests and source-aware scanner artifacts bind to its immutable digest; independent review finds no confirmed bypass/regression. Production adoption remains a separate decision.
**Tests:** XML ingestion/chunking suites, font discovery, drawtext/subtitles and existing FFmpeg capability controls; source/SBOM consistency and current scanner runs.
**Status:** Not Started

- [ ] Integrate the qualified artifacts into a separately named candidate image, without changing production recipes. Run `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_xml_ingestion.py` and Chunking tests `test_xml_allows_url_text.py`, `test_json_xml_offsets.py`, `test_xml_tail_preservation.py` against that image.
- [ ] Run fontconfig discovery and FFmpeg drawtext/subtitle controls; compare the previously accepted FFmpeg capability inventory without broadening accepted retirements.
- [ ] Save source packages, Python source/SBOM, build/configuration logs, binary hashes, package controls, test statuses and candidate OCI identity. Run existing pinned Syft/Trivy/Grype tooling; explicitly account for bundled Expat even if automatic catalogers omit it. Do not rewrite findings into passes.
- [ ] Perform the security-fix skill's single fresh read-only candidate review after focused gates, confirm any concrete findings, rerun affected checks, and update TASK-13013.7.9 with exact evidence and remaining blockers.

## Source evidence

- Debian CVE status and required follow-up: https://security-tracker.debian.org/tracker/CVE-2026-66046
- Expat signed tag object `6dd46f3d0fcaeda0af6d00fd496acb8fe96b900d`, target commit `12cf0b1f25f026a022fe728ad8f7e3d017285b80`; GitHub reports the tag signature valid. Release digest independently matches the downloaded archive: https://github.com/libexpat/libexpat/releases/tag/R_2_8_4
- CPython's exact vendoring procedure: https://github.com/python/cpython/blob/v3.12.14/Modules/expat/refresh.sh
- Official Python image source digest/configure precedent: https://github.com/docker-library/python/blob/master/3.12/slim-trixie/Dockerfile (recorded 2026-09-07; the recipe in this plan pins the archive, not moving master).

The original candidate's immutable manifest is `94b8ed9fa76fdc3959c7b58a5b2417277bb2171d2d29803f33ac6e593d1c7942`. Its system package is 2.8.3 and its pyexpat binary statically embeds `expat_2.8.3`; this is not a claim that current production has been remediated.
