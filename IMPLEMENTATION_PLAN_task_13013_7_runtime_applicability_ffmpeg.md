# Exact-image applicability and FFmpeg candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Persist stronger runtime evidence and evaluate an authenticated FFmpeg security-remediation candidate without weakening release admission; only the explicitly approved legacy `pp` capability may be retired.

**Architecture:** Extend the existing isolated Python runtime probe, which CI already binds to the exact OCI candidate. Evaluate FFmpeg separately from production packaging, using signed upstream source, an immutable source hash, the existing Debian feature inventory, synthetic compatibility probes, and a fresh scan.

**Tech Stack:** Python, pytest, Docker/BuildKit, Debian trixie, FFmpeg, GnuPG, Trivy.

**Spec:** User approval on 2026-09-06 of the proposal recorded in TASK-13013.7.3: test a signed, hash-pinned FFmpeg 9.0.1 build; preserve current media capabilities; add persistent exact-image Chroma UUID isolation and OS applicability checks. Tracking: TASK-13013.7.5 and TASK-13013.7.6.

**Approved amendment:** The requester subsequently approved retiring `pp`/libpostproc and building a pinned newer libplacebo to continue the FFmpeg 9 candidate. That approval does not permit any other unassessed capability loss or production promotion before verification.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design`; do not change the shared checkout/environment or unrelated sessions' work.
- Retain Chroma. No vulnerability exceptions, allowlists, weakened thresholds, or automatic CVE disposition from runtime evidence.
- Production Dockerfiles and release image selection remain unchanged during candidate evaluation.
- The only approved removed capability is the `pp` filter/libpostproc. Preserve or explicitly report every other baseline inventory delta; do not suppress vulnerability findings.
- Use task-owned temporary data and containers; runtime probes have no network, capabilities, host secrets, or writable host data mounts.
- Activate the project virtual environment before Python, pytest or Bandit. The task-owned PyJWT environment is `/private/tmp/task-13013-7-pyjwt-verification-venv`; shared dependencies are read-only. Host Chroma is not the locked image version.
- Test first, run scoped Bandit and formatting, and obtain independent review. Stop and reassess after three failed attempts per issue.
- Each implementer owns only its assigned files, must not spawn agents, and leaves commits to the controller unless explicitly delegated.

## Stage 1: Persist exact-image applicability evidence
**Goal:** Stronger evidence using existing OCI-bound CI probes.
**Success Criteria:** Foreign UUID reads and writes fail closed, original data survives, and bounded OS facts are recorded without waiver claims.
**Tests:** Runtime probe unit/integration suites, locked-image synthetic check, CI contract tests, Bandit.
**Status:** Complete

Current-head `4415afe0d4` CI run `34044783356` passed all three runtime isolation checks in app, worker and audio-worker. Artifact checksums, OCI subject digests and scanner config identities verified. This completes evidence collection, not vulnerability disposition.

Implementation and scoped independent review are complete in commit `98886152ed`: 56 tests passed with four existing warnings; Black/compileall passed; scoped Bandit found no unreviewed findings. The final local-image probe passed with all three installed versions matching the lock. Publication/rebuilt current-head OCI evidence is pending; this is distinct from the supporting local-image result.

### Task 1: Extend the existing runtime probe (TASK-13013.7.5)

**Files:** Modify `Helper_Scripts/Supply_Chain/runtime_probe.py`, `tldw_Server_API/tests/Supply_Chain/test_runtime_probe.py`, `tldw_Server_API/tests/Supply_Chain/test_runtime_probe_integration.py`; add a short evidence-scope paragraph in `Docs/Development/Software_Supply_Chain.md`.

**Interfaces:** Preserve `probe_chroma(*, proc_net: Path | None = None) -> str` and existing evidence fields. Extend the successful checks object with `foreign_collection_uuid_isolation: true`. Add `os_facts` from `probe_os_facts() -> dict`; no admission code consumes it.

- [x] Write failing tests that inject a client accepting a foreign collection UUID, a client throwing an unrelated exception, and a client corrupting the original collection; each must fail rather than emit successful evidence. Test OS facts with absent executable, successful observation, malformed output, and command failure. Assert emitted evidence distinguishes missing facilities from unobserved/failed checks.
- [x] Run the new tests and record the expected failures before implementation.
- [x] In `probe_chroma`, after existing positive collection/query tests, attempt each client's low-level operations against the other collection UUID in both directions. Accept only `chromadb.errors.NotFoundError`; successful calls and other errors fail the probe. Keep synthetic records and no listener checks. Use the following operations, then verify both IDs AND original documents remain unchanged:

```python
operations = (
    ("_get", {}),
    ("_count", {}),
    ("_query", {"query_embeddings": [[1.0, 0.0]], "n_results": 1}),
    ("_add", {"ids": ["intruder"], "embeddings": [[1.0, 0.0]]}),
    ("_update", {"ids": [foreign_record_id], "documents": ["modified"]}),
    ("_upsert", {"ids": ["intruder"], "embeddings": [[1.0, 0.0]]}),
    ("_delete", {"ids": [foreign_record_id]}),
)
# getattr(client, operation)(collection_id=foreign_collection.id, **arguments)
# Require NotFoundError for every call; then get(include=["documents"]).
```

- [x] Implement a small bounded OS-fact collector using fixed executable paths and subprocess argument arrays (never a shell), `check=True`, captured output and a timeout. Record Perl `ivsize` and `ptrsize` from `/usr/bin/perl -V:ivsize -V:ptrsize` when available, and file-presence booleans for `/usr/lib/systemd/systemd-homed` and `/lib/systemd/systemd-homed`. Missing Perl is an explicit absent observation, not 64-bit evidence. Failure or malformed Perl evidence fails closed. Do not install diagnostic dependencies or infer that all Perl/systemd CVEs are non-applicable. Broader module/function reachability and frontend evidence are outside this bounded collector.
- [x] Add the OS facts to successful `probe` JSON and retain the existing scope string: `isolated embedded-client probe; not application startup or a vulnerability waiver`.
- [x] Run `source /private/tmp/task-13013-7-pyjwt-verification-venv/bin/activate && python -m pytest tldw_Server_API/tests/Supply_Chain/test_runtime_probe.py tldw_Server_API/tests/Supply_Chain/test_runtime_probe_integration.py -q`; run applicable existing CI contract tests and scoped Bandit on the touched probe. Format changed Python only. Document host versus exact-image evidence precisely.
- [x] Return a report with changed files, red/green commands/results, Bandit findings, self-review, and remaining exact-image verification needs. Controller performs locked image execution and independent review before committing.

## Stage 2: Build an authenticated compatibility candidate
**Goal:** A separately tagged FFmpeg candidate with authenticated source and explicit feature comparison.
**Success Criteria:** Signature/hash verification precedes compilation; capabilities and representative operations are measured against the current image; production selection stays unchanged.
**Tests:** Source identity rejection tests; build, feature inventory, synthetic audio/video probes.
**Status:** Complete

The prior `pp`/libplacebo compatibility stop is lifted by explicit requester approval. Build with libplacebo 7.360.1, retain OpenGL/Vulkan support, and document the approved `pp` retirement. The removed libglslang switch is replaced by upstream external SPIR-V compiler detection, not by dropping Vulkan filters. Production packaging remains unchanged.

Build investigation amendment: system Vulkan headers lack the KHR surface/swapchain maintenance symbols needed by libplacebo. Use its exact upstream `3rdparty/Vulkan-Headers` gitlink `450bd2232225d6c7728a4108055ac2e37cef6475`; archive `https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/450bd2232225d6c7728a4108055ac2e37cef6475`, SHA-256 `26df9841c30806a994e2fdf42f7c87bcb1ced9db9a06033469123939fb3fa075`. Extract into that submodule directory before Meson; the headers and registry must stay synchronized. This supersedes the system-Vulkan-header assumption below, without adding another runtime library or unpinned downloads.

### Task 2: Authenticate source and evaluate the candidate (TASK-13013.7.6)

**Files:** Create `Dockerfiles/candidates/ffmpeg/Dockerfile`, `Dockerfiles/candidates/ffmpeg/README.md`, `Helper_Scripts/Supply_Chain/ffmpeg_candidate.py`, and `tldw_Server_API/tests/Supply_Chain/test_ffmpeg_candidate.py`. No production Dockerfile, dependency lock, admission policy, existing runtime probe, or unrelated source edits.

**Interfaces:** Consumes the current FFmpeg `-buildconf`, codec/filter/protocol inventories and official upstream release identity. Produces a candidate image ID, source SHA-256, signature verification evidence, exact configure arguments, feature deltas, and compatibility results; it does not produce a production replacement.

**Verified source identity:** `ffmpeg-9.0.1.tar.xz` SHA-256 `cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635`; detached RSA signature validated against the published fingerprint on 2026-09-06. Evidence: `/private/tmp/task-13013-7-ffmpeg-candidate.08RU6j/source-verification-gnupg.txt`. Baseline inventory is in `baseline-capabilities.txt` in the same directory. First configuration attempt rejected the removed `--disable-omx` switch; omit this already-disabled legacy option without omitting any enabled feature. Debian trixie supplies libplacebo 7.349.0 while this source requires >=7.351.0; validate and record this dependency constraint before proposing any packaging promotion.

**Pinned inputs and build contract:**

- FFmpeg archive URL: `https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz`; hash above. Signature URL appends `.asc`, SHA-256 `b613a00005232a1245ace7080088781ac23a916119d3e5b0d6c042368eee0177`. Key URL `https://ffmpeg.org/ffmpeg-devel.asc`, SHA-256 `397b3becedcd5a98769967ff1ff8501ddc89f8368b8f766e4701377d7dbaabe5`; fingerprint `FCF986EA15E6E293A5644F10B4322F04D67658D8`. Verify all hashes and the detached signature in a fresh temporary GnuPG home before extracting or building.
- libplacebo version `7.360.1`, signed tag object `719cc95244a1f1d648dd72459822e026e6530f22`, commit `cee9b076f2c63104ccfd497fa79c39a867293ec4`. Download `https://codeload.github.com/haasn/libplacebo/tar.gz/cee9b076f2c63104ccfd497fa79c39a867293ec4`; archive SHA-256 `6f8fa218cbafd8e5f50b8a82d918e1d8bbb92f9f980820bc0b34d92e9b79e484`. GitHub API reports this upstream maintainer tag signature valid; saved evidence `libplacebo-tag.json` in the temporary evidence directory. Describe this precisely, not as local detached-signature verification of the archive.
- Use the existing pinned production-compatible base `python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea` for a multi-stage candidate-only recipe. The build stage may enable Debian source repositories and install Trixie FFmpeg build dependencies plus `gnupg`, `curl`, `meson`, `ninja-build`, `python3-glad`, `python3-jinja2`, `libfast-float-dev`, `libxxhash-dev`, and `glslang-tools` as needed. No Sid/Forky binary packages and no unpinned submodule downloads. Use system Python for Meson. Configure Meson with `--wrap-mode=nodownload`, explicit enabled OpenGL/Vulkan/glslang/lcms and disabled demos; use Trixie system GLAD/Vulkan headers/fast_float rather than fetching submodules.
- Install both projects under `/opt/tldw-ffmpeg9`, with shared libraries in its `lib` directory. Configure FFmpeg with rpath to that prefix, all existing baseline enabled features from the captured build configuration, omit removed `--disable-omx`, and replace removed `--enable-libglslang` with installed external `glslangValidator` compilation. Do not remove another enabled feature just to make configuration succeed. Preserve ffmpeg, ffprobe and ffplay. Prefer a fixed explicit configure argument list in the recipe, not a runtime-derived policy.
- Keep compilers/build dependencies out of the candidate runtime. Derive the exact installed Debian runtime package/version requirements from the built binaries/libraries' dynamic dependencies, verify no missing libraries or old libav ABI appears, and install that explicit package set in the runtime stage. Do not install the Debian ffmpeg package alongside the candidate or silently delete package metadata to hide findings. Include the source identities and selected runtime package versions as candidate evidence. A source-built binary that the scanner cannot identify is a coverage gap, not a clean result.
- The controller will run Docker builds and fresh scans; the implementer writes the recipe/helper/tests/docs and self-reviews. No commits, pushes, Backlog mutations or subagents from the implementer.

**Minimal helper/test contract:** Use Python standard library only. Provide source SHA-256 verification over actual files, capability inventory parsing/comparison, and a candidate CLI for reproducible inventory plus synthetic media probes. Keep this task-specific, not a generic build or policy engine. A function `verify_sha256(path: Path, expected: str) -> None` must reject modified/missing files and malformed hashes. A function `parse_capabilities(output: str, category: str) -> set[str]` must parse real FFmpeg listings including aliases and separate input/output protocols. A comparison must allow missing `pp` only in filters and report every other removed capability; malformed/empty inventory must not become success. Preserve raw inventories alongside parsed results.

Test first with literal sample listings and synthetic files: unchanged hash passes, tampered/missing content fails; normal encoder/decoder/filter/muxer/protocol rows parse correctly; empty or malformed listing fails; only `pp` removal passes; removed mp3/opus/h264/libplacebo or a protocol is reported. Exercise real helper behavior, not source-text grep or fake executable success. Synthetic run uses a temporary directory and fixed argument-array subprocesses with timeouts, verifies output metadata/decoded data rather than only exit codes, and covers WAV resampling, MP3/FLAC/Opus/AAC round trips, H264/AAC MP4, thumbnail, segment/concat. Document that no real GPU/device capability is proven by software-only probes.

- [x] Download official `https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz`, its `.asc`, and `https://ffmpeg.org/ffmpeg-devel.asc` into task-owned temporary storage. Verify detached signature with the independently published fingerprint `FCF986EA15E6E293A5644F10B4322F04D67658D8`; calculate SHA-256 only after valid signature. If official source retrieval/authentication fails, stop candidate compilation and report that specific blocker.
- [x] Capture baseline `ffmpeg -buildconf`, `-encoders`, `-decoders`, `-demuxers`, `-muxers`, `-filters`, `-protocols` from task-owned image `sha256:e90381ffb6a7a8f57783c11d54265d5ddcd740fc763118f043088d441ffa1f54` with network disabled. Preserve its feature set; removed/renamed upstream features must be explicitly assessed rather than silently omitted.
- [x] Before dispatching the recipe implementation, add the verified source hash and complete build configuration/dependency requirements to this task. The controller is responsible for investigation, not inventing source identity. Use a pinned Debian-compatible builder and an isolated installation prefix to avoid overwriting system libav ABI. Fail closed for signature/hash mismatch, missing enabled dependencies, or failed compilation.
- [x] Write and run failing source-identity/configuration tests; implement the smallest candidate-only recipe and tests. Build into a task-owned Docker tag. No package deletion to hide findings, no production image change.
- [x] Compare baseline versus candidate inventories and exercise synthetic WAV resampling, MP3/FLAC/Opus/AAC audio encoding/decoding, H.264 video plus audio to MP4, ffprobe metadata, thumbnail extraction, and segment/concat behavior. Treat unexplained capability losses as blockers, not successful remediation.

## Stage 3: Scan, review and record disposition
**Goal:** Evidence-backed next action without claiming unsupported remediation.
**Success Criteria:** Candidate scans are tied to its immutable identity; all remaining blockers are explicitly recorded.
**Tests:** Pinned fresh Trivy scan, scoped pytest/Bandit, independent review.
**Status:** In Progress

### Task 3: Validate evidence and document the decision (TASK-13013.7.5 / TASK-13013.7.6)

**Files:** Existing plan and Backlog records via CLI; candidate findings documentation from Task 2.

**Interfaces:** Consumes runtime JSON and candidate identity/inventories/probe output. Produces a reviewed evidence record and next action; never changes scanner exceptions or release gates.

- [x] Execute the extended Chroma probe in the task-owned locked-Chroma image using a read-only script mount, no network/capabilities and temporary synthetic databases; distinguish this check from subsequently rebuilt exact release images.
- [x] Scan the candidate with `ghcr.io/aquasecurity/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969` and a database updated within 24 hours. Preserve SBOM, source component identity, scan JSON and image/config identity. A scanner that does not identify the source-built FFmpeg is a coverage gap, not a clean result.
- [ ] Map FFmpeg findings to authoritative upstream/Debian fix evidence, retaining all unresolved findings. Separate OS facts, applicability conclusions and remediation results in documentation.
- [ ] Independently review task diffs and final combined changes, fix regressions with covering tests, and record verification and blockers via Backlog CLI. Commit only verified code; production promotion and PR merge remain gated by actual compatibility/security results and existing human summary policy.

## Verified candidate checkpoint — 2026-09-06

The candidate-only build succeeded. Immutable local image/index identity: `sha256:b1946062bcca4120ed619e8f3c9a0d7546f816b343831a27f080dc285b3562ba`; platform manifest: `sha256:8c5566434d2ac2019ca383517dda5fa7cbbb5ac3916ff2fe74e22a1a29492e32`; config: `sha256:5a8d1fd5d6bade011a137d322883b1809530d179a2a2eb062ea4fd71db2aadbd`. The scanner config identity matches the build. This is not a published release artifact.

All 179 derived runtime package versions match their builder selections; the full runtime has 275 installed packages, versus 785 recorded builder packages. Runtime dependency checks found no unresolved libraries, old FFmpeg ABI packages, or compiler packages. `gcc-14-base` is runtime support, not the compiler. The actual evaluator ran as UID/GID 65534 with no network or capabilities, three read-only input-file mounts, and container-owned output storage.

All synthetic media probes passed. The evaluator correctly returned exit 1 and `compatible: false` for these additional upstream retirements; only `pp` is currently approved:

| Surface | Additional missing names | Assessment |
| --- | --- | --- |
| Encoders/decoders | `sonic`, `sonicls` (encoder only) | Upstream removed the Sonic implementation; no retirement approval yet. |
| Encoders/decoders | `v308`, `v408`, `v410` | Explicit codec names retired in favor of raw pixel formats. Baseline-generated MOV files decoded identically in supporting diagnostic tests. |
| Input protocol | `hls` | Old protocol handler removed; ordinary local HLS demuxing/decoding passed a supporting diagnostic test. |
| Output devices | `opengl`, `sdl`, `sdl2` | Upstream retired these outputs; `ffplay` remains present, but this is not transparent CLI compatibility. |

Upstream evidence: [Sonic removal](https://github.com/FFmpeg/FFmpeg/commit/ca1d253621d4643b7b9ba13ded6dfc68c329fda5), [V4xx removal](https://github.com/FFmpeg/FFmpeg/commit/0438966d84deb34394df1929a53ab79fee5cc8d4), [HLS protocol removal](https://github.com/FFmpeg/FFmpeg/commit/64fafd63f0b4ebf8dbbdbdc2296f21a03548b5fc), [SDL output removal](https://github.com/FFmpeg/FFmpeg/commit/17b31065f0f74711882ddd26c5ae7a15fdd910a8), [OpenGL output removal](https://github.com/FFmpeg/FFmpeg/commit/9283c5251f6a96d09b12e1b344437c885d673464). Application/helper/frontend searches found no direct use of these names; this is not proof for downstream callers.

The pinned offline Trivy scan used a database updated at `2026-09-06T13:03:04Z`, roughly four hours before scanning. It reports 73 High/Critical OS package findings covering 37 distinct CVEs. The SBOM contains 277 components, but neither source-built FFmpeg nor libplacebo is identified. This is a coverage blocker, not evidence that FFmpeg findings are resolved. Source identity manifests and separate static build-input package evidence are retained; no package metadata, exceptions, or thresholds were changed to hide findings.

The recipe records exact selected Debian package versions but does not pin a Debian repository snapshot; later rebuilds are not guaranteed to select identical build inputs. This reproducibility gap remains explicit before promotion. Full authoritative FFmpeg CVE disposition and source-built scanner coverage remain unfinished.

Independent task review and two scoped correction rounds are clean. Final controller verification: 255 tests passed, one opt-in Bun/Docker skip, four existing warnings; Black/Ruff clean; Bandit zero findings. The helper verifies the fixed source hash; genuine detached-signature verification belongs to the build recipe, not substring checks over saved logs. A final whole-branch merge review remains pending the unresolved release blockers.

Evidence directory: `/private/tmp/task-13013-7-ffmpeg-candidate.08RU6j`, including `clean-image-build-evidence/`, `clean-runtime-evaluation/candidate-evaluation.json`, `trivy-clean-ffmpeg9.json`, `sbom-clean-ffmpeg9.cdx.json`, build logs, and final test/Bandit reports. Production packaging and PR merge remain blocked. The requester has been asked to choose additional upstream interface retirements versus compatibility-preserving 7.x security backports; no additional retirement is assumed.
