# Exact-image applicability and FFmpeg candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Persist stronger runtime evidence and evaluate an authenticated FFmpeg security-remediation candidate without weakening release admission or reducing media capabilities.

**Architecture:** Extend the existing isolated Python runtime probe, which CI already binds to the exact OCI candidate. Evaluate FFmpeg separately from production packaging, using signed upstream source, an immutable source hash, the existing Debian feature inventory, synthetic compatibility probes, and a fresh scan.

**Tech Stack:** Python, pytest, Docker/BuildKit, Debian trixie, FFmpeg, GnuPG, Trivy.

**Spec:** User approval on 2026-09-06 of the proposal recorded in TASK-13013.7.3: test a signed, hash-pinned FFmpeg 9.0.1 build; preserve current media capabilities; add persistent exact-image Chroma UUID isolation and OS applicability checks. Tracking: TASK-13013.7.5 and TASK-13013.7.6.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design`; do not change the shared checkout/environment or unrelated sessions' work.
- Retain Chroma. No vulnerability exceptions, allowlists, weakened thresholds, or automatic CVE disposition from runtime evidence.
- Production Dockerfiles and release image selection remain unchanged during candidate evaluation.
- Use task-owned temporary data and containers; runtime probes have no network, capabilities, host secrets, or writable host data mounts.
- Activate the project virtual environment before Python, pytest or Bandit. The task-owned PyJWT environment is `/private/tmp/task-13013-7-pyjwt-verification-venv`; shared dependencies are read-only. Host Chroma is not the locked image version.
- Test first, run scoped Bandit and formatting, and obtain independent review. Stop and reassess after three failed attempts per issue.
- Each implementer owns only its assigned files, must not spawn agents, and leaves commits to the controller unless explicitly delegated.

## Stage 1: Persist exact-image applicability evidence
**Goal:** Stronger evidence using existing OCI-bound CI probes.
**Success Criteria:** Foreign UUID reads and writes fail closed, original data survives, and bounded OS facts are recorded without waiver claims.
**Tests:** Runtime probe unit/integration suites, locked-image synthetic check, CI contract tests, Bandit.
**Status:** In Progress

Implementation and scoped independent review are complete in commit `98886152ed`: 56 tests passed with four existing warnings; Black/compileall passed; scoped Bandit found no unreviewed findings. The final local-image probe passed with all three installed versions matching the lock. Publication/rebuilt current-head OCI evidence is pending; this is distinct from the supporting local-image result.

### Task 1: Extend the existing runtime probe (TASK-13013.7.5)

**Files:** Modify `Helper_Scripts/Supply_Chain/runtime_probe.py`, `tldw_Server_API/tests/Supply_Chain/test_runtime_probe.py`, `tldw_Server_API/tests/Supply_Chain/test_runtime_probe_integration.py`; add a short evidence-scope paragraph in `Docs/Development/Software_Supply_Chain.md`.

**Interfaces:** Preserve `probe_chroma(*, proc_net: Path | None = None) -> str` and existing evidence fields. Extend the successful checks object with `foreign_collection_uuid_isolation: true`. Add `os_facts` from `probe_os_facts() -> dict`; no admission code consumes it.

- [ ] Write failing tests that inject a client accepting a foreign collection UUID, a client throwing an unrelated exception, and a client corrupting the original collection; each must fail rather than emit successful evidence. Test OS facts with absent executable, successful observation, malformed output, and command failure. Assert emitted evidence distinguishes missing facilities from unobserved/failed checks.
- [ ] Run the new tests and record the expected failures before implementation.
- [ ] In `probe_chroma`, after existing positive collection/query tests, attempt each client's low-level operations against the other collection UUID in both directions. Accept only `chromadb.errors.NotFoundError`; successful calls and other errors fail the probe. Keep synthetic records and no listener checks. Use the following operations, then verify both IDs AND original documents remain unchanged:

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

- [ ] Implement a small bounded OS-fact collector using fixed executable paths and subprocess argument arrays (never a shell), `check=True`, captured output and a timeout. Record Perl `ivsize` and `ptrsize` from `/usr/bin/perl -V:ivsize -V:ptrsize` when available, and file-presence booleans for `/usr/lib/systemd/systemd-homed` and `/lib/systemd/systemd-homed`. Missing Perl is an explicit absent observation, not 64-bit evidence. Failure or malformed Perl evidence fails closed. Do not install diagnostic dependencies or infer that all Perl/systemd CVEs are non-applicable. Broader module/function reachability and frontend evidence are outside this bounded collector.
- [ ] Add the OS facts to successful `probe` JSON and retain the existing scope string: `isolated embedded-client probe; not application startup or a vulnerability waiver`.
- [ ] Run `source /private/tmp/task-13013-7-pyjwt-verification-venv/bin/activate && python -m pytest tldw_Server_API/tests/Supply_Chain/test_runtime_probe.py tldw_Server_API/tests/Supply_Chain/test_runtime_probe_integration.py -q`; run applicable existing CI contract tests and scoped Bandit on the touched probe. Format changed Python only. Document host versus exact-image evidence precisely.
- [ ] Return a report with changed files, red/green commands/results, Bandit findings, self-review, and remaining exact-image verification needs. Controller performs locked image execution and independent review before committing.

## Stage 2: Build an authenticated compatibility candidate
**Goal:** A separately tagged FFmpeg candidate with authenticated source and explicit feature comparison.
**Success Criteria:** Signature/hash verification precedes compilation; capabilities and representative operations are measured against the current image; production selection stays unchanged.
**Tests:** Source identity rejection tests; build, feature inventory, synthetic audio/video probes.
**Status:** In Progress

Candidate promotion is blocked: authenticated FFmpeg 9.0.1 omits the baseline `pp`/libpostproc filter. Preserving every existing capability therefore requires a revised approach, not merely configure-flag adjustments. No candidate compilation, successful compatibility run or fresh binary scan is claimed. A newer libplacebo is also required, and the old libglslang enable switch has been replaced by external SPIR-V compiler detection. Production packaging is unchanged.

### Task 2: Authenticate source and evaluate the candidate (TASK-13013.7.6)

**Files:** Candidate-only recipe and verification material under `Dockerfiles/candidates/ffmpeg/`; focused tests under `tldw_Server_API/tests/Supply_Chain/`; candidate findings under `Docs/Development/`. No production Dockerfile edits.

**Interfaces:** Consumes the current FFmpeg `-buildconf`, codec/filter/protocol inventories and official upstream release identity. Produces a candidate image ID, source SHA-256, signature verification evidence, exact configure arguments, feature deltas, and compatibility results; it does not produce a production replacement.

**Verified source identity:** `ffmpeg-9.0.1.tar.xz` SHA-256 `cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635`; detached RSA signature validated against the published fingerprint on 2026-09-06. Evidence: `/private/tmp/task-13013-7-ffmpeg-candidate.08RU6j/source-verification-gnupg.txt`. Baseline inventory is in `baseline-capabilities.txt` in the same directory. First configuration attempt rejected the removed `--disable-omx` switch; omit this already-disabled legacy option without omitting any enabled feature. Debian trixie supplies libplacebo 7.349.0 while this source requires >=7.351.0; validate and record this dependency constraint before proposing any packaging promotion.

- [x] Download official `https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz`, its `.asc`, and `https://ffmpeg.org/ffmpeg-devel.asc` into task-owned temporary storage. Verify detached signature with the independently published fingerprint `FCF986EA15E6E293A5644F10B4322F04D67658D8`; calculate SHA-256 only after valid signature. If official source retrieval/authentication fails, stop candidate compilation and report that specific blocker.
- [x] Capture baseline `ffmpeg -buildconf`, `-encoders`, `-decoders`, `-demuxers`, `-muxers`, `-filters`, `-protocols` from task-owned image `sha256:e90381ffb6a7a8f57783c11d54265d5ddcd740fc763118f043088d441ffa1f54` with network disabled. Preserve its feature set; removed/renamed upstream features must be explicitly assessed rather than silently omitted.
- [ ] Before dispatching the recipe implementation, add the verified source hash and complete build configuration/dependency requirements to this task. The controller is responsible for investigation, not inventing source identity. Use a pinned Debian-compatible builder and an isolated installation prefix to avoid overwriting system libav ABI. Fail closed for signature/hash mismatch, missing enabled dependencies, or failed compilation.
- [ ] Write and run failing source-identity/configuration tests; implement the smallest candidate-only recipe and tests. Build into a task-owned Docker tag. No package deletion to hide findings, no production image change.
- [ ] Compare baseline versus candidate inventories and exercise synthetic WAV resampling, MP3/FLAC/Opus/AAC audio encoding/decoding, H.264 video plus audio to MP4, ffprobe metadata, thumbnail extraction, and segment/concat behavior. Treat unexplained capability losses as blockers, not successful remediation.

## Stage 3: Scan, review and record disposition
**Goal:** Evidence-backed next action without claiming unsupported remediation.
**Success Criteria:** Candidate scans are tied to its immutable identity; all remaining blockers are explicitly recorded.
**Tests:** Pinned fresh Trivy scan, scoped pytest/Bandit, independent review.
**Status:** Not Started

### Task 3: Validate evidence and document the decision (TASK-13013.7.5 / TASK-13013.7.6)

**Files:** Existing plan and Backlog records via CLI; candidate findings documentation from Task 2.

**Interfaces:** Consumes runtime JSON and candidate identity/inventories/probe output. Produces a reviewed evidence record and next action; never changes scanner exceptions or release gates.

- [ ] Execute the extended Chroma probe in the task-owned locked-Chroma image using a read-only script mount, no network/capabilities and temporary synthetic databases; distinguish this check from subsequently rebuilt exact release images.
- [ ] Scan the candidate with `ghcr.io/aquasecurity/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969` and a database updated within 24 hours. Preserve SBOM, source component identity, scan JSON and image/config identity. A scanner that does not identify the source-built FFmpeg is a coverage gap, not a clean result.
- [ ] Map FFmpeg findings to authoritative upstream/Debian fix evidence, retaining all unresolved findings. Separate OS facts, applicability conclusions and remediation results in documentation.
- [ ] Independently review task diffs and final combined changes, fix regressions with covering tests, and record verification and blockers via Backlog CLI. Commit only verified code; production promotion and PR merge remain gated by actual compatibility/security results and existing human summary policy.
