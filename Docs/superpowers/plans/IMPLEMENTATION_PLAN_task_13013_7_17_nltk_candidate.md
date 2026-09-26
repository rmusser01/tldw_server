# NLTK candidate backport implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Qualify a source-verifiable NLTK 3.10.3 candidate containing the approved model-path fixes and their necessary sandbox prerequisites.

**Architecture:** Retain immutable upstream inputs and one explicit adapted backport patch. Use standard wheel tooling in separate disposable environments, real behavioral controls, and existing pinned scanner patterns. This is a candidate experiment, not a new dependency-management framework.

**Tech Stack:** Python stdlib, setuptools/wheel, pytest, existing Docker/scanner tools.

**Spec:** `Docs/Design/TASK-13013.7.17-nltk-candidate-backport.md`

## Global Constraints

- Work only in `codex/task-13013-7-supply-chain-design` at `.worktrees/task-13013-7-supply-chain-design`.
- Associated Backlog task: TASK-13013.7.17. Keep its notes current through the official CLI.
- Do not modify production Dockerfiles, pyproject.toml, uv.lock, Bun locks, application source, exception records, release workflows or another task's artifacts. No publication or merge.
- The user approved the three official fixes plus the concrete prerequisite expansion in the spec. No unrelated ZIP, network, tokenizer or package-resource behavior changes.
- After the focused review reproduced an out-of-root directory creation, the user approved the minimal five-caller adaptation to consume validated returned paths. Record this as a local correction, not verbatim upstream code.
- Preserve upstream version `3.10.3`; candidate wheel build tag is `1tldw1`. No official fixed-release claim. Every qualification result retains `admitted:false`.
- Real security controls use synthetic files only under a unique owned temporary directory, with an allowed data root and sibling forbidden canaries. No real secrets, user files or external exploit targets.
- Do not mutate shared virtual environments. Activate `/private/tmp/task-13013-7-pyjwt-verification-venv/bin/activate` before local Python tooling; it inherits NLTK 3.9.1 and is only suitable for repository helper tests. Create new isolated environments for wheel qualification.
- Fail closed on source/hash/patch drift, unexpected changed files, provenance mismatch or failed controls. Never suppress a vulnerability match to make the experiment pass.
- Tests first; retain RED/GREEN commands and outcomes. Run focused Black/Ruff, compile and Bandit checks on executable touched scope. Track test-assert B101 separately from executable findings.
- One implementation agent at a time; agents do not spawn other agents. Each stage gets independent spec/quality review. No FFmpeg source-history investigation and no reading another plan's SDD directory.
- Maximum three unsuccessful attempts per issue, then reassess the cause and alternatives; do not blindly retry.

## Stage 1: Verified backport source

**Goal:** Produce an explicit, reviewable patch that applies exactly to authenticated released source.
**Success Criteria:** Required helper closure is present; unchanged files are unchanged; source/patch substitutions and unsafe archives are rejected.
**Tests:** Source-preparation rejection cases plus actual-source compile/import smoke checks in a fresh environment.
**Status:** Complete

### Task 1: Prepare the pinned source and adapted patch

**Files:**
- Create `Dockerfiles/candidates/nltk/source-inputs.json`, `backport.patch`, `prepare.py`, `README.md`.
- Create `tldw_Server_API/tests/Supply_Chain/test_nltk_candidate_sources.py`.
- Introduce `Dockerfiles/candidates/nltk/controls.py` with focused changing-path and legitimate PathLike regressions for the review correction; Task 2 extends this same harness.
- Update the design with final immutable source identities and precise selected/omitted hunk mappings.

**Interfaces:**
- Consume the spec's exact source/wheel URLs and SHA256 values from uv.lock; original patch identities are in the spec.
- `prepare.py --inputs DIR --output DIR` consumes downloaded local inputs, validates them against `source-inputs.json`, applies the reviewed patch, and emits `source/` plus `source-provenance.json` into a new output directory. It does not download, install or publish anything.
- `source-provenance.json` records schemaVersion 1, scope `nltk-candidate-source-not-release-admission`, admitted false, input filenames/URLs/SHA256, backport SHA256, and before/after SHA256 for every changed source file.
- Use `prepare(inputs: Path, output: Path) -> dict` as the tested entry point. No generic plugin/registry/class hierarchy.

- [x] Verify the retained inputs in `/private/tmp/nltk-3103-inputs.DBRgMb`; obtain any missing immutable upstream raw file with its commit URL and SHA256 before selecting code. Never trust the suffix `deb6` without resolving its full commit from the research report or official API.
- [x] Write real preparation tests before implementation. Test local source/patch mutations, unexpected source file edits, missing patch inputs, duplicate/path-traversal/symlink/hardlink archive entries, pre-existing output, and a successful small controlled fixture. Example boundary assertion:

```python
def test_rejects_existing_output_without_modifying_it(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises((ValueError, FileExistsError)):
        prepare(tmp_path / "inputs", output)
    assert sentinel.read_bytes() == b"keep"
```

- [x] Run the focused test file and retain the expected missing-helper/incorrect-behavior RED results.
- [x] Build one adapted patch against the exact released sdist, preserving upstream code and license notices. Required source changes: primary fix runtime hunks for transitionparser, picklesec, maxent, named_entity, perceptron, crf, hunpos and stanford; prior bf511 picklesec denylist; data staging helper family; pathsec tool-path helper family and constants/imports, required validate_path corrections, and deferred O_TRUNC. Preserve original primary patches as hashed external inputs rather than copying their unrelated test suites wholesale into the installed wheel.
- [x] Source helper selection must come from an exact immutable upstream revision, not manually redesigned guard logic. Record each function/hunk's source. Preserve the smallest import/constant closure; do not copy full develop data.py/pathsec.py. If a new functional adaptation is required beyond the approved boundary, report its concrete reason before making it.
- [x] Implement preparation with stdlib hashing/JSON/tarfile and checked subprocess argv. Validate every archive member before extraction. Use a fixed package root, reject collisions and links, apply the local patch without fuzz, then verify an exact source inventory against the known changed-file set and expected hashes. Keep failure evidence outside the source tree. Do not execute downloaded source during extraction.

```python
with tarfile.open(archive, "r:gz") as source:
    members = source.getmembers()
    # Validate the full member list before extracting any member.
    # Only regular files and directories under nltk-3.10.3 are permitted.
```

- [x] Run focused tests, actual source preparation, compile-only checks, and an import smoke check in a fresh private environment with recorded dependencies. This smoke check proves import closure, not vulnerability remediation.
- [x] Self-review the complete adapted diff against immutable inputs, document a44 context adaptation and omitted upstream tests, run lint/Bandit, and commit the stage with its Backlog/design/plan changes. Report source output path and exact manifest interface for Task 2.

## Stage 2: Installed-wheel behavioral qualification

**Goal:** Build the exact candidate and demonstrate the corrected model boundaries without breaking legitimate model operations.
**Success Criteria:** All selected security controls reject safely and all legitimate controls pass against installed wheels; baseline failures are retained.
**Tests:** Actual TransitionParser, maxent/named entity, perceptron and shared guard controls; installed RECORD/module identity checks.
**Status:** In Progress

Blocked checkpoint: the platform rejected the next security-test step before
an installed environment or candidate wheel was built. Qualification remains
incomplete. Do not retry the rejected test via another agent, tool, or phrasing.
Production and admission remain unchanged.

Static-only checkpoint: commit `41ee97fd640e9d5df97baf201ab3619025e80e0e`
contains the artifact verifier, pinned qualification inputs, and 14 synthetic
tests. The receiving session independently reviewed the static commit on
2026-09-09 and found three parsing gaps: normalized ZIP path aliases, repeated
singleton identity headers, and missing/unsupported wheel format versions.
These are ordinary static-verifier corrections within the handoff's scope.
This work does not satisfy installed-wheel controls or complete Task 2.

Static follow-up tracking:
- [x] Independently review `648017aecb..41ee97fd64`; retain the review in this
  plan's workspace as `task-2-static-review-2026-09-09.md`.
- [x] Recheck all 22 requirements against retained wheel SHA256 and exactly one
  top-level METADATA name/version, including joblib's cloudpickle requirement.
  Audit JSON: `/private/tmp/task-13013-7-17-static-review-vd9bf6ls/requirements-audit.json`,
  SHA256 `34767d214bd0b5f7debe7843d1ac99b486525d45c3b9044ab1402e52d0fef1a7`.
- [x] Correct the three findings with failing synthetic fixtures, passing
  controls, and independent scoped re-review. RED: 15 failed / 14 passed;
  GREEN: 29 passed. Scoped review approved all three fixes with no new
  significant breakage; reports are `task-2-static-fix-report-2026-09-09.md`
  and `task-2-static-rereview-2026-09-09.md` in this plan's workspace.
- [x] Run final focused provenance tests, Black/Ruff, compile-only and Bandit.
  Controller: 29 tests passed with five existing warnings; Black/Ruff and two
  compile-only checks passed. Bandit: verifier zero findings/errors; tests 13
  Low-severity/high-confidence B101 assertion findings, no suppressions/errors.
  Raw final logs: `/private/tmp/task-13013-7-17-static-review-vd9bf6ls/`
  (`pytest-final.log`, `bandit-verifier.json`, `bandit-tests.json`).

Warnings retained: Starlette/httpx deprecation, unknown pytest `plugins` option,
the intentional duplicate ZIP fixture warning, passlib's `crypt` deprecation,
and an existing Pydantic field shadow. No unrelated configuration or warning
suppression changes. The full Supply_Chain suite was not rerun at this checkpoint.

No candidate wheel, package installation/execution, runtime qualification or
scanner evidence is produced by this static follow-up. AC2/AC3 remain open.

Subsequent 2026-09-09 regression checkpoint at code commit `862078fcab`:
the full default Supply_Chain suite passed 715 tests, with five explicit
Docker/image opt-in skips and five warnings, in 31.67 seconds; process exit 0.
Independent repository-path preflight found no execution of the restricted
NLTK runtime controls or FFmpeg history investigation. This suite includes
other local integration checks. The shared tool environment uses pytest 8.4.1,
below the declared dev minimum 9.0.3, so supported-environment qualification
remains open. Evidence, exact scope, warning ownership and live advisory status:
`Docs/Evidence/TASK-13013.7-remaining-status-2026-09-09.md`.

### Task 2: Build and exercise separate baseline/candidate wheels

**Files:**
- Extend `Dockerfiles/candidates/nltk/controls.py`; create `wheel-provenance.py`, `requirements-qualification.txt`.
- Create `tldw_Server_API/tests/Supply_Chain/test_nltk_wheel_provenance.py`.
- Update `Dockerfiles/candidates/nltk/README.md`.

**Interfaces:**
- Consume Task 1 `source/` and `source-provenance.json`.
- `controls.py --evidence DIR` runs from an installed environment with explicit owned test paths; writes `controls.json` with per-case outcome and runtime identity, admitted false. Return nonzero when any expected guard or positive control fails; baseline nonzero is retained as RED evidence, never silently recast as passing.
- `wheel-provenance.py --wheel FILE --source DIR --source-provenance FILE --output FILE` verifies wheel members and RECORD against the prepared source and input record. Writes schemaVersion 1, admitted false, wheel SHA256, metadata version/build and verified module hashes.

- [ ] Pin qualification dependencies with full hashes in an experiment-only requirements file. Prefer applicable versions already in uv.lock; use official immutable wheels for any necessary build/test-only dependency absent there. Required runtime families: NLTK dependencies, numpy/scipy/scikit-learn, python-crfsuite for positive CRF cases, pytest, build/setuptools/wheel. Never modify uv.lock or the shared environment.
- [ ] Use isolated Linux/amd64 containers from the existing worker base `python:3.11.16-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534`. The host cannot allocate a named semaphore (errno28), causing joblib serial fallback despite free disk space. Do not change host semaphore limits or clear shared resources. Record the Linux/aarch64 Docker daemon and emulated amd64 execution honestly; no native-performance certification. Give each environment its own owned output, mount inputs read-only, publish no ports, and disable networking during controls.
- [ ] Write provenance rejection tests for changed wheel modules, missing/incorrect RECORD, duplicate/unsafe members, wrong version/build tag, unexpected source-file changes and mismatched source-provenance hashes. Verify a small hand-derived fixture passes.
- [ ] Before building/testing the candidate wheel, run real behavioral controls against the official baseline wheel: both TransitionParser train/parse using a tiny real trained model; AveragedPerceptron save/load; PerceptronTagger save/load round trip; maxent parameter save/load; private default staging; malformed, outside-root, traversal, symlink and hardlink writes with unchanged canary bytes. Use process-level timeouts for special-file cases; no unbounded blocking tests.

```python
canary.write_bytes(b"outside-model-canary")
os.link(canary, allowed / "linked-model")
try:
    with pathsec.open(allowed / "linked-model", "w") as stream:
        stream.write("must not escape")
except (PermissionError, ValueError):
    pass
assert canary.read_bytes() == b"outside-model-canary"
```

- [ ] Build using standard setuptools/wheel tooling in the isolated environment, e.g. the backend's `bdist_wheel --build-number 1tldw1`. Keep NLTK VERSION/METADATA at 3.10.3. Record build interpreter/tool artifacts, commands, source inputs and wheel hash. Do not claim reproducibility from a single build.
- [ ] Install baseline and candidate separately, from local exact wheels with network disabled for controls. Verify loaded nltk module paths are inside each environment and hashes match installed RECORD/prepared source before behavioral assertions. Set a controlled NLTK data path and ENFORCE true; do not weaken the sandbox to get positive controls passing.
- [ ] Launch controls with process-local `TMPDIR=/tmp` on this POSIX host and allocate the test workspace under `/private/tmp`. NLTK trusts private platform temp directories on macOS, so explicitly verify the sibling canary is outside every effective allowed root before testing. Never copy the upstream fixture's home-directory writes or its `enforce_off` fixture. Keep all synthetic artifacts in the owned test workspace and preserve real guard implementations.
- [ ] Run the identical control suite on both; candidate must pass, baseline results distinguish observed failures from existing passing controls. Add selected upstream denylist tests and CRF normal/path-rejection controls. Explicitly label Stanford/Hunpos mocked external-process coverage as argument-boundary coverage, not actual JVM/binary qualification.
- [ ] Run repository provenance tests/lint/Bandit, self-review, document all unsupported optional runtimes and skipped tests, then commit. Return both environment paths, exact wheel locations/hashes, and retained RED/GREEN evidence to Task 3.

## Stage 3: Compatibility and vulnerability evidence

**Goal:** Bind actual application compatibility and complete scanner results to the tested candidate.
**Success Criteria:** Resource-backed RAG/chunking controls pass with downloads disabled; full scan evidence and unresolved advisory status are retained honestly.
**Tests:** Actual resource-backed query and semantic operations, evidence identity rejection tests, same-database scanner validation.
**Status:** Not Started

### Task 3: Qualify application consumers and record the complete result

**Files:**
- Create `Dockerfiles/candidates/nltk/application-controls.py`, `evidence.py`.
- Create `tldw_Server_API/tests/Supply_Chain/test_nltk_candidate_evidence.py`.
- Create `Docs/Evidence/TASK-13013.7.17-nltk-candidate-qualification.md`.
- Update `Dockerfiles/candidates/nltk/README.md` and Backlog task.

**Interfaces:**
- Consume exact Task 2 wheels/environments, wheel provenance and controls JSON.
- `application-controls.py --repository DIR --resources DIR --evidence DIR` imports real application query_features and semantic strategy with candidate NLTK. Writes `application-controls.json` with source/module/resource hashes and observed behavior; nonzero on failure.
- `evidence.py --evidence DIR` reads retained source/wheel/control/scan evidence, verifies identity and database equality, and emits `qualification.json` with schemaVersion 1, admitted false and qualified scope only. It must not suppress findings or change release policy.

- [ ] Use exact, hashed official resource artifacts for Punkt/Punkt tab, stopwords and WordNet as required by actual consumers. Retain resource commit/URL/hash and inspect archives before extraction. No runtime download and no using ambient user corpora.
- [ ] Write an application control that runs actual query tokenization/WordNet expansion and semantic sentence spans using real resources. Confirm the resource-backed branch ran through observed loaded resources/callable provenance, not just that fallback gave plausible output. Include expected sentence offsets and deterministic known tokens; keep app source unchanged.
- [ ] Verify the evidence aggregator rejects mixed wheel identities, missing control output, failed positive controls, mismatched database hashes and incomplete scans. Example critical invariant:

```python
assert report["admitted"] is False
assert report["candidateWheelSha256"] == verified_candidate_sha256
assert report["scannerRecognitionIsRemediationProof"] is False
```

- [ ] Run SBOM/scanners over exact baseline/candidate installed artifacts using existing pinned versions: Syft v1.51.1, Trivy 0.74.0, Grype v0.118.0 (digests in frontend-runtime-candidate.yml). Retain full JSON and tool identity, inventory/wheel provenance, current database metadata and before/after SHA256, using one frozen fresh database per scanner for both artifacts. Follow the existing candidate workflow's scan pattern; no new CI workflow required for this experiment.
- [ ] Record every remaining NLTK advisory match at version 3.10.3 and complete dependency findings. No dropped matches, waiver, renamed package, inferred fixed claim or production admission. If upstream now has a release, verify it separately without substituting inputs mid-experiment.
- [ ] Main controller runs fresh covering verification, independent final review of this task's commit range, and records factual results in the durable evidence doc and Backlog. Remove only this plan file after all stages complete, retaining required review history until integration. No publication, production adoption or merge follows from this qualification.
