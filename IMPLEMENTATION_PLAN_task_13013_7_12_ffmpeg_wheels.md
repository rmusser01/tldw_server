# Candidate FFmpeg wheel remediation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans for inline execution, or subagent-driven-development if the user chooses delegation. Execute task-by-task with review checkpoints; the parent owns security-fix changes. Use fix-finding for security implementation and its required independent investigation/review.

**Goal:** Rebuild the two owning wheels against verified repaired FFmpeg 8.x and qualify a new immutable candidate without changing production or release policy.

**Architecture:** A source gate binds all original scanner inputs to complete repairs and pinned build inputs. Native wheel builders produce two private-library wheels; a derivative assembler replaces only those distributions. Same-image controls and source-aware scans produce evidence, not automatic release admission.

**Tech Stack:** Python standard library, pytest, native Linux amd64, existing pinned uv/Buildx tooling, upstream PyAV/OpenCV build systems, Syft/Trivy/Grype, GitHub Actions.

**Spec:** [Approved design](Docs/Design/2026-09-07-task-13013-7-12-ffmpeg-wheel-remediation.md).

## Global constraints

- Work only in `.worktrees/task-13013-7-supply-chain-design`, branch `codex/task-13013-7-supply-chain-design`, under TASK-13013.7.12. Do not edit another worktree or task's repository.
- Preserve `av==18.1.0`, `opencv-python==5.0.0.93`, dependency requirements, features and private-library packaging. Distinguish rebuilt wheels by build tags and hashes, not false upstream versions.
- Production Dockerfiles, `pyproject.toml`, `uv.lock`, release admission, scanner thresholds, ignore lists and VEX remain unchanged. No registry publishing.
- Keep the existing standalone FFmpeg 9 recipe and signed Debian snapshot inputs unchanged. No headless substitution, dependency deletion, codec/filter/hardware disabling, global loader overrides or bare-library swaps.
- Preserve all 43 original component matches across 22 CVE IDs. A source repair and an applicability verdict are distinct; no automatic exception follows from either.
- Preserve original candidate archive SHA-256 `864b2e19887641bbe2f6505d0b28f44ccec3cb1cc3d1de4f65063f5211c1c478` and subject `08f4a090041b1d87d779e1436073910c0b6c4afc2ffcb9a6d957a94c307b45bb` as immutable inputs.
- Native execution only for qualification. Runtime controls: UID/GID 10001, no network, read-only root, no capabilities, bounded scratch/CPU/memory/PIDs. No host application/dependency overlays.
- No expensive build until the source gate is complete. No qualification with missing tests, skips, timeouts, unresolved findings/evidence, unknown source identity or unsupported hardware claims.
- Use apply_patch for edits, Backlog MCP/CLI for tracking, an activated virtual environment for Python tools. Reassess after three failed attempts at one issue.

## File map and existing contracts

Create files incrementally, only with the task that uses them:

| Candidate-local file | Responsibility |
| --- | --- |
| `Dockerfiles/candidates/ffmpeg-wheels/source-lock.json` | Verified source/tool/asset identities, ordered patches and per-input coverage |
| `Dockerfiles/candidates/ffmpeg-wheels/source-inputs.py` | Validate manifest, input bytes and coverage before build |
| `Dockerfiles/candidates/ffmpeg-wheels/patches/` | Reviewed upstream patch bytes with provenance |
| `Dockerfiles/candidates/ffmpeg-wheels/README.md` | Source acquisition, reproduction, evidence limits and license records |
| `Dockerfiles/candidates/ffmpeg-wheels/Dockerfile` | Native offline source/wheel build stages; no runtime deployment |
| `Dockerfiles/candidates/ffmpeg-wheels/wheel-evidence.py` | Wheel metadata, RECORD, ELF ownership/dependency and capability evidence |
| `Dockerfiles/candidates/ffmpeg-wheels/derivative.Dockerfile` | Replace only the two wheels in exact retained image |
| `Dockerfiles/candidates/ffmpeg-wheels/runtime-inventory.py` | Before/after file ownership and unchanged-runtime gate |
| `Dockerfiles/candidates/ffmpeg-wheels/qualify.py` | Native behavioral and mapped-library controls |
| `Dockerfiles/candidates/ffmpeg-wheels/scan-evidence.py` | Bind raw scans/source SBOMs to image and source/wheel inventories |
| `.github/workflows/candidate-ffmpeg-wheels.yml` | Opt-in acquisition, native builds, derivative controls and retained evidence |

Tests: `test_ffmpeg_wheel_sources.py`, `test_ffmpeg_wheel_evidence.py`,
`test_ffmpeg_wheel_inventory.py`, `test_ffmpeg_wheel_qualification.py`, and
`test_ffmpeg_wheel_scans.py` in `tldw_Server_API/tests/Supply_Chain/`.
Use that directory's importlib-based helper-loading convention, not a new package.

Read/reuse, without weakening or changing original producer pins:

- `compatibility/retained-inputs.py`: `verify_inputs(metadata: Path, payloads: dict[str, Path]) -> dict` and `verify_loaded(inspection: list[dict]) -> str` apply to the **old input only**.
- `combined-expat/assembly-inputs.py`: `verify_metadata(run: dict, artifact: dict, producer: dict) -> None` and `verify_payload(path: Path, expected: str) -> None`.
- `combined-expat/image-identity.py`: `verify(path: Path, loaded_config: str, commit: str) -> dict`; distinguish config, manifest and subject. Layer payload validation is separately required.
- `.github/workflows/candidate-compatibility.yml`, `compatibility/application-provenance.py`, `combined-expat/run-tests.py`, and `Helper_Scripts/Supply_Chain/ffmpeg_candidate.py` define existing controls. Do not point the old retained-subject validator at a new derivative.

All candidate-local paths above are relative to `Dockerfiles/candidates/` when abbreviated in the existing-contract list.

## Stage 1: Verified source ledger

**Goal:** Establish complete, reproducible repair inputs before any compilation.
**Success Criteria:** All 43 input identities accounted for; actual fixes and prerequisites verified; authenticated baseline, owning sources and build closure pinned; no unresolved source-gate entries.
**Tests:** Tampered input, duplicate/missing coverage, partial patch series, wrong source identity, unsafe archive paths and unauthenticated claims are rejected.
**Status:** In Progress

### Task 1: Resolve and retain the exact repair inputs

**Files:** Create source-lock, patches and README from the file map; update this plan and Backlog with verified identities. Do not create speculative hashes or an apparently ready lock with unknown values.

**Interfaces:** `source-lock.json` schema `ffmpeg-wheel-source/v1` contains `inputs`, `patches`, `coverage`, `builds`, and `evidence`. Each input has `name`, official `url`, `sha256`, `authentication` evidence and local `path`. Each ordered patch has upstream full `commit`, `sha256`, `path`, `requires`, and affected `source_paths`. Coverage records retain `input_id`, `cve`, `owner`, `source_paths`, `disposition` (`repaired`, `already_fixed`, `absent_condition`), `repair_commits`, `evidence_paths`, and `regression_id`. Dispositions describe source state, not scanner admission. Build records contain owning version, ABI/platform, exact tools/assets and configuration arguments. Evidence paths are confined relative paths with hashed input records.

- [ ] Recover `wheel-triage.json`, raw Grype/Syft/Trivy reports and upstream records from `/private/tmp/task-13013-7-combined-scans.vpkHOr`. Verify the original 43 unique identities and raw Grype SHA-256 `2bba9bbad78ea5f0cb3dc2c8d3e645185a83f8c4a18665e1bb323965913dd194`; retain a hash manifest and durable task artifact. Never overwrite raw evidence.
- [ ] Resolve the official signed FFmpeg 8.1 release archive and verify signature/signer against official published material. Record archive/source identity before extraction. Retain the inspected maintenance snapshot only as comparison evidence, not as complete remediation.
- [ ] Inspect upstream history for floodfill's allocation fix, hqdn3d's earlier rejection guard and MagicYUV. Check full repairs/prerequisites for all other entries. For RSCC use repair `a5fe21a1a410a680fe93c33b0dd696b7e1c3aea4` / backport `cd1f545cf27ba08f6f5b31b1e92665d7874d4fd7`; for CineForm use `16b2049d4d5222db6cd7c031409058571c94f6a9` / `db05df9d135fb56a4babb836d5e9f5c1d984e087`, after verifying baseline context. Do not apply both equivalent repairs or their feature-introduction references.
- [ ] Build a source-condition comparison for every CVE on the chosen baseline. Fetch official patch bytes, check dependencies, apply in a fresh source tree, and hash all changed sources. Reverse checks alone are insufficient. Retain original/repaired source diffs and bounded upstream regression fixtures; report hardware test requirements explicitly.
- [ ] Inspect exact PyAV/OpenCV source packaging, submodules, original wheel ELF closure/configuration, tools and native build dependencies. Acquire from official sources; pin every archive, tool image and required offline test asset. Preserve licensing and capabilities. If supported build requirements cannot preserve the approved versions/features, stop with evidence for design revision.
- [ ] Commit only verified source inputs and evidence metadata: `docs(supply-chain): pin wheel repair inputs (TASK-13013.7.12)`. Record any unresolved condition in Backlog and leave Stage 1 incomplete; do not manufacture a completed source-lock.

### Task 2: Fail-closed source gate

**Execution note:** The deterministic validator and synthetic fixture tests are
being implemented independently while Task 1's real input ledger remains
incomplete. This does not authorize a build or replace the real source
integration and independent evidence review below. No real `source-lock.json`
is emitted by this implementation slice.

**Files:** Create `source-inputs.py` and `test_ffmpeg_wheel_sources.py`; extend README.
**Interfaces:** `verify_sources(lock: dict, root: Path, original_matches: list[dict]) -> dict` raises `ValueError` on malformed or incomplete evidence and returns `{schema_version, source_lock_sha256, input_sha256, coverage_ids}` only after all checks. CLI takes `--lock`, `--root`, `--original-matches`; emits JSON only on success.

- [ ] Write tests first using small real hashed fixture files. Load the helper with importlib as in `test_candidate_compatibility.py`. Include this behavioral assertion (fixture `source_case` holds a valid lock/root/original-match set):

```python
def test_missing_component_coverage_fails(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"].pop()
    with pytest.raises(ValueError, match="coverage"):
        helper.verify_sources(lock, root, matches)
```

- [ ] Run `python -m pytest -q tldw_Server_API/tests/Supply_Chain/test_ffmpeg_wheel_sources.py` in the activated task venv. Confirm failure is missing helper/behavior, not unrelated fixture setup.
- [ ] Implement strict schema/types, unique identity and exact coverage-set checks; SHA-256 verification of regular non-symlink inputs; containment checks; ordered prerequisite validation; exact versions/ABI; authentication-evidence requirements. Reject unknown dispositions, missing regression IDs, absolute/traversing paths, duplicate patch application and extra unreviewed files. Do not fetch or execute external programs in this validator.
- [ ] Add parametrized negative tests for each rejection and positive tests for all three source dispositions. Confirm forged signer text without evidence is rejected; the trusted acquisition step owns authenticity, the validator owns byte/metadata binding. Verify patch application in an isolated upstream source integration test, not only synthetic records.
- [ ] Run focused tests, full Supply_Chain tests, Ruff/Black and Bandit on the touched implementation/tests; classify existing pytest B101 assertions separately. Review the source ledger independently, then commit `feat(supply-chain): gate wheel source inputs (TASK-13013.7.12)`.

## Stage 2: Deterministic owning-wheel builds

**Goal:** Produce complete, identifiable replacement wheels without reduced functionality.
**Success Criteria:** Native builds use only verified inputs; original package versions/requirements retained; private repaired libraries and complete source/license/build evidence; per-wheel capability comparison passes.
**Tests:** Offline acquisition boundary, wrong ABI/version, corrupt RECORD, duplicate/escaping archive entries, unbound native dependency and capability loss failures.
**Status:** Not Started

### Task 3: Native build and wheel evidence

**Files:** Create candidate Dockerfile, `wheel-evidence.py`, `test_ffmpeg_wheel_evidence.py` and workflow; extend README. Source-lock's verified build entries from Stage 1 supply exact commands/flags and hashes; no live resolver is allowed to invent this closure.
**Interfaces:** `verify_wheel(path: Path, expected: dict, native_inventory: dict) -> dict` returns `{owner, version, wheel_sha256, build_tag, files, native_libraries, capabilities, source_lock_sha256}`. `files` maps wheel-relative paths to SHA-256. `native_libraries` records path/hash/SONAME/NEEDED/source IDs. `expected` comes from verified source-lock; `native_inventory` comes from pinned native ELF tools and runtime baseline probes, not wheel-supplied assertions.

- [ ] Write a minimal wheel ZIP fixture and rejection tests before helper implementation:

```python
def test_changed_metadata_version_is_rejected(wheel_case, helper):
    wheel, expected, inventory = wheel_case
    expected["version"] = "0.0.0"
    with pytest.raises(ValueError, match="version"):
        helper.verify_wheel(wheel, expected, inventory)
```

- [ ] Run the focused test file and observe RED. Implement metadata/RECORD/archive integrity checks, package requirements/build-tag/ABI checks and reviewed closure/capability comparison. Reject extraneous files, collisions, RECORD mismatches, traversal, old library hashes and libraries without source bindings. Test a valid repaired wheel fixture as well as every failure class.
- [ ] Add separate native build targets for PyAV/OpenCV using their verified upstream build commands. Acquire inputs in a bounded network-enabled stage; require source gate success before offline compilation. Use `RUN --network=none` for compile/build/repair phases. Preserve compiler/configure logs and wheel repair output. No build tools in the derivative runtime.
- [ ] Configure opt-in workflow dispatch with `contents: read` and `actions: read`, pinned actions/images, native architecture checks, finite timeout, task-specific concurrency, and evidence upload on failure. No publish step or production workflow edits. Derive input artifact IDs and hashes from fixed reviewed records, not untrusted arbitrary URLs.
- [ ] Verify both generated wheels natively, compare each original wheel's capabilities and licenses, and repeat the offline build to compare payload hashes. Record reproducibility differences; do not claim bit-for-bit reproducibility unless measured. Normalize only documented build metadata, never hide binary differences.
- [ ] Run helper/unit/workflow lint and touched-scope security checks; review; commit `build(supply-chain): rebuild candidate FFmpeg wheels (TASK-13013.7.12)`. Update Backlog with immutable run/artifact identities; do not advance on missing hardware/source evidence.

## Stage 3: Isolated derivative and native controls

**Goal:** Replace only the two owning distributions and prove same-image behavior/library provenance.
**Success Criteria:** New attested identity; no old wheel libraries; all unrelated runtime paths unchanged; required media/application controls pass without skips.
**Tests:** Ownership ambiguity, unrelated mutation, stale library, wrong image identity, import-order collision, changed media content and missing test phases all fail.
**Status:** Not Started

### Task 4: Owned-file replacement and immutable identity

**Files:** Create `runtime-inventory.py`, `derivative.Dockerfile`, `test_ffmpeg_wheel_inventory.py`; extend workflow/README.
**Interfaces:** `verify_transition(before: dict, after: dict, old_owned: set[str], new_owned: set[str]) -> dict` returns `{unchanged_paths, removed_owned_paths, installed_owned_paths}`. Inventories map absolute image paths to `{kind, sha256, target, mode, uid, gid}`; directories omit content hashes. Ownership sets come from validated package RECORD plus separately verified distribution-owned metadata, never an arbitrary caller glob. Inventory collection records files, symlinks, modes and ownership, including relevant empty directories.

- [ ] Write RED tests with small dictionaries:

```python
def test_unrelated_runtime_change_fails(helper):
    before = {"/opt/libpython.so": {"kind": "file", "sha256": "a" * 64}}
    after = {"/opt/libpython.so": {"kind": "file", "sha256": "b" * 64}}
    with pytest.raises(ValueError, match="unrelated"):
        helper.verify_transition(before, after, set(), set())
```

- [ ] Implement ownership/collision checks, exact outside-union equality, removal of superseded owned files and hash agreement with wheel evidence. Add tests for symlink target/mode changes, stale library paths, namespace overlap and package metadata ambiguity. Permit no broad path exemption for site-packages or system directories.
- [ ] Verify original retained artifact through fresh API metadata and reviewed hashes before loading. Reuse old input gate only here. Stage complete wheels and exact installer outside runtime, then install offline with dependencies disabled and only the two named distributions forcibly replaced. Capture inventories before/after; fail on any unreviewed change. Do not sync the original lock afterward because that would reinstall upstream wheels.
- [ ] Export a fresh attested OCI archive, hash every referenced blob, bind source/wheel/producer identities, and establish subject/manifest/config correspondence before execution. Check derivative platform/user independently. Never relabel the old subject as the new image or use its hard-coded validator to approve the derivative.
- [ ] Run focused/full helper tests and touched-scope lint/security checks; review; commit `build(supply-chain): isolate wheel candidate replacement (TASK-13013.7.12)`.

### Task 5: Actual media consumers and loaded libraries

**Files:** Create `qualify.py`, `test_ffmpeg_wheel_qualification.py`; extend workflow/README.
**Interfaces:** `validate_report(report: dict, expected: dict) -> None` requires an exact `required_controls` set with successful exits/no skips and exact mapped-file hashes from wheel evidence. Report includes `image_subject`, `config_digest`, `wheel_sha256`, per-control `status`, behavioral measurements and mapped-library observations. Driver isolates import orders in separate processes and exits nonzero on any control failure; it cannot convert missing resources into skips.

- [ ] Write RED validator tests, including missing controls and an old loaded library:

```python
def test_old_mapped_library_fails(qualification_case, helper):
    report, expected = qualification_case
    report["mapped_libraries"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mapped"):
        helper.validate_report(report, expected)
```

- [ ] Implement native controls using deterministic tone/image/video fixtures. Check audio sample count, channel count/rate, duration and bounded lossy-codec error; test PCM/WAV/MP3/FLAC/Opus/AAC with actual PyAV, Faster-Whisper decode and TTS writer. Bind thresholds to the original wheel baseline before derivative comparison; do not widen thresholds to pass.
- [ ] Exercise RapidOCR image preparation/color/resize and pinned offline OCR model controls; exercise OpenCV's original supported video reader/writer backend and check decoded pixels/frame count. Missing model assets/backend support is an explicit failure, not skipped coverage.
- [ ] Run PyAV alone, OpenCV alone, PyAV then OpenCV, and OpenCV then PyAV. Capture `/proc/self/maps` inside each exercised process after operations, hash mapped files and compare against built wheel evidence. Check linked closure with pinned ELF tooling; do not rely solely on `ldd`, version strings or imports.
- [ ] Re-run original eight application tests/24 phases and import observer on the derivative; verify original Python/Expat binary hashes, rendering and unchanged standalone FFmpeg evaluator/baseline. Include Stage 1 bounded security regressions; require hardware-backed evidence for claims that need hardware. Preserve nonfatal warnings without declaring them repaired.
- [ ] Run negative subprocess controls for timeout/crash/missing phase/skip/bad media output, then native positive controls. Run regression suite/lint/Bandit; review; commit `test(supply-chain): qualify rebuilt wheel consumers (TASK-13013.7.12)`.

## Stage 4: Source-aware scans and review

**Goal:** Bind complete raw security evidence to the new candidate and report remaining blockers accurately.
**Success Criteria:** Native exact-subject scans and source inventories retained; all original inputs reconciled individually; independent review; no suppression or automatic admission.
**Tests:** Wrong subject, modified raw output, missing scanner/DB metadata, missing bundled/static source identity or discarded original match fails evidence validation.
**Status:** Not Started

### Task 6: Evidence binding and final review

**Files:** Create `scan-evidence.py`, `test_ffmpeg_wheel_scans.py`; extend workflow/README and update Backlog.
**Interfaces:** `verify_scan_bundle(bundle: dict, expected: dict, root: Path) -> dict` returns `{scope, image_subject, raw_sha256, original_input_ids, unresolved_ids}`. `scope` is always `candidate-evidence-not-admitted`; `expected` includes immutable derivative/source/wheel hashes. Every raw report, database record, source SBOM and reconciliation record is a hashed, confined file beneath `root`.

- [ ] Write RED tests with real small report fixtures:

```python
def test_wrong_scan_subject_fails(scan_case, helper):
    bundle, expected, root = scan_case
    bundle["image_subject"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="subject"):
        helper.verify_scan_bundle(bundle, expected, root)
```

- [ ] Implement exact subject/raw-file/DB/source-closure validation and individual 43-input reconciliation. Reject dropped duplicates, absent SBOM components and unsupported claims of complete coverage. Keep scanner execution success separate from vulnerability clearance; unresolved matches cannot silently become admitted.
- [ ] Run native pinned Syft/Trivy/Grype with fresh recorded database identities. Start from previously verified scanner versions/digests in TASK-13013.7.9 and reverify official availability before use. Preserve raw reports, all error logs and resource-limit settings; do not use ignore/VEX flags or edited reports. Include source-derived identities for wheel/static native dependencies and retain unrelated Expat/libplacebo/glslang/SPIR-V coverage gaps rather than claiming this task repairs them.
- [ ] Obtain the security-fix skill's fresh review of source-to-binary repair and regressions plus compatibility/patch-risk review. Fix valid findings with RED/GREEN tests; rerun affected gates against a new identity whenever binary contents change.
- [ ] Run final verification commands below, record exact outcomes/run links/hashes in Backlog and commit `test(supply-chain): bind wheel security evidence (TASK-13013.7.12)`. Mark criteria complete only when evidenced. If hardware or source evidence is unavailable, leave the relevant gate open and report it.
- [ ] Hand off a candidate-only result. Do not merge PR #2869 until its separate release blockers, current-dev rebase, review comments and human Change summary gate are satisfied. Do not claim remediation of unrelated package findings. Remove only this task's implementation plan once all stages are complete, preserving the approved design and durable final record.

## Verification and checkpoints

From this isolated worktree, activate the existing task environment before Python:

```sh
source /private/tmp/task-13013-7-pyjwt-verification-venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Supply_Chain
python -m ruff check Dockerfiles/candidates/ffmpeg-wheels tldw_Server_API/tests/Supply_Chain/test_ffmpeg_wheel_*.py
python -m black --check Dockerfiles/candidates/ffmpeg-wheels tldw_Server_API/tests/Supply_Chain/test_ffmpeg_wheel_*.py
python -m compileall -q Dockerfiles/candidates/ffmpeg-wheels
python -m bandit -r Dockerfiles/candidates/ffmpeg-wheels -f json -o /private/tmp/bandit_task_13013_7_12.json
actionlint .github/workflows/candidate-ffmpeg-wheels.yml
git diff --check
```

Run commands only after their task has created the referenced files. Test-file
Bandit must also be run and assertion-only B101 findings distinguished from other
findings, without globally disabling a rule. No commit-hook bypasses. Select only
task files for each commit. At each stage record verification and remaining gates
in Backlog; a local helper pass does not replace native evidence. Preserve failing
artifacts and retry with fresh run identities rather than overwriting evidence.

## Plan review

- Source authentication, complete repairs, retained input coverage and durable evidence: Stage 1.
- Owning versions, private-library ABI, native build closure, capability/license preservation: Stage 2.
- Restricted replacement, unrelated-runtime equality, new immutable identity, real consumers and original controls: Stage 3.
- Source-aware raw scans, individual reconciliation, independent review, unchanged admission and recovery: Stage 4.
- User selected subagent-driven execution. Stage 1 investigation is active; no candidate build has started.
