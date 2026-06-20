# VZ Smoke Evidence Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the host-side VZ Linux smoke wrapper emit the canonical evidence bundle consumed by sandbox operator status.

**Architecture:** Keep evidence generation inside `run-host-e2e-smoke.sh` so the same command that proves host execution also produces operator-consumable diagnostics. The wrapper records bounded phase metadata, expected sidecar files, runtime paths, bundle hashes, cleanup state, and a final server env export without requiring the API server to scrape logs or accept request-supplied paths.

**Tech Stack:** Bash smoke wrapper, Python/pytest shell-level tests, existing sandbox operator evidence JSON contract.

**Baseline discovery:** Current `dev` already includes default evidence directory handling, expected sidecars, JSON evidence generation, private directory checks, and failure-preserving cleanup evidence. This slice is therefore narrowed to the missing operator handoff contract: make the wrapper print a sourceable `TLDW_SANDBOX_VZ_EVIDENCE_DIR` export and document how to use it.

---

### Task 1: Add Evidence Bundle Tests

**Files:**
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`

- [x] **Step 1: Write dry-run option coverage**

Add a test that runs `run-host-e2e-smoke.sh --dry-run --evidence-dir <path>` and asserts:
- help mentions `--evidence-dir PATH`
- dry-run output includes `TLDW_SANDBOX_VZ_EVIDENCE_DIR=<path>`
- dry-run does not create the evidence directory

- [x] **Step 2: Write fake-run success evidence coverage**

Add a test with fake helper and fake pytest runner that completes successfully and asserts:
- `<evidence_dir>/host-smoke-evidence.json` exists
- JSON includes `schema_version=1`, `final_exit_code=0`, `evidence_dir`, runtime paths, skip flags, and phase statuses
- expected sidecars exist: `source-bundle-hashes-before.txt`, `source-bundle-hashes-after.txt`, `run-bundle-hashes.txt`, `runtime-paths.txt`, `cleanup-status.txt`

- [x] **Step 3: Write fake-run failure evidence coverage**

Add a test where fake pytest fails one phase and asserts:
- wrapper exits non-zero
- `host-smoke-evidence.json` still exists
- `final_exit_code` is non-zero
- failed phase is recorded with non-zero `exit_code`
- cleanup sidecar is still written

- [x] **Step 4: Run tests and verify RED**

Run:

```bash
source ../../.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q --tb=short
```

Result: focused tests failed on the missing `export TLDW_SANDBOX_VZ_EVIDENCE_DIR=...` line in dry-run and real-run output; existing bundle emission coverage was already present.

### Task 2: Implement Wrapper Evidence Emission

**Files:**
- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`

- [x] **Step 1: Add option and defaults**

Add:
- `EVIDENCE_DIR="${DEFAULT_RUNTIME_DIR}/evidence"`
- `--evidence-dir PATH`
- usage text documenting that the path should be private if reused across runs

- [x] **Step 2: Add phase runner helper**

Replace direct phase calls with a helper that:
- prints/runs the command
- captures exit code without aborting immediately
- records `status`, `exit_code`, and UTC timestamp in shell variables
- preserves existing command output behavior

- [x] **Step 3: Add evidence directory preparation**

For real runs:
- reject symlink evidence dirs
- reject non-directory existing paths
- create missing dir as `0700`
- require owner-private permissions

For dry-run:
- print intended `TLDW_SANDBOX_VZ_EVIDENCE_DIR` without creating directories

- [x] **Step 4: Write sidecars and JSON**

At cleanup/finalization:
- write bundle hashes before/after using portable `shasum -a 256` or `sha256sum`
- write run-bundle hashes against the same bundle path for this slice
- write runtime path and cleanup status sidecars
- write `host-smoke-evidence.json` with bounded JSON fields using Python from `PYTHON_BIN`
- print `export TLDW_SANDBOX_VZ_EVIDENCE_DIR=<path>` after writing evidence

- [x] **Step 5: Preserve failure semantics**

Ensure the wrapper:
- still exits non-zero when a phase fails
- still attempts helper cleanup
- still writes evidence on failed phases
- does not delete or mutate non-socket socket paths

### Task 3: Document Operator Wiring

**Files:**
- Modify: `tools/vz-linux-image/README.md`

- [x] **Step 1: Update Helper Smoke section**

Document:
- default evidence path under the runtime dir
- optional `--evidence-dir PATH`
- the printed `export TLDW_SANDBOX_VZ_EVIDENCE_DIR=...`
- starting/restarting the API server with that env var to show evidence in operator status

- [x] **Step 2: Keep docs copy/paste-safe**

Preserve the existing `trap 'rm -rf "${runtime_dir}"' EXIT` cleanup pattern.

### Task 4: Validate And Commit

**Files:**
- Modify: Backlog task `TASK-2393`

- [ ] **Step 1: Run focused tests**

```bash
source ../../.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q --tb=short
```

- [ ] **Step 2: Run shell/static checks**

```bash
bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
git diff --check
```

- [ ] **Step 3: Run Bandit for touched Python tests if required**

```bash
source ../../.venv/bin/activate && python -m bandit -r tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -f json -o /tmp/bandit_vz_smoke_evidence_bundle.json
```

- [ ] **Step 4: Update Backlog and commit**

Record verification in `TASK-2393`, then commit the wrapper, tests, docs, plan, and task.
