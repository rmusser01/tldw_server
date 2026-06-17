# VZ Host Smoke Evidence Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default-on structured evidence output to the VZ Linux host smoke wrapper.

**Architecture:** Keep evidence capture inside `run-host-e2e-smoke.sh`, because both local operator runs and the host-gated workflow already delegate to it. The wrapper writes concise files under a private evidence directory and the existing workflow artifact upload retains them automatically.

**Tech Stack:** Bash wrapper, Python/pytest wrapper tests, GitHub Actions workflow contract tests, Markdown policy docs.

---

## File Structure

- Modify `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`: add `--evidence-dir`, safe directory preparation, phase tracking, evidence file generation, and exit-code-preserving trap behavior.
- Modify `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`: add portable fake-helper tests for default evidence behavior, override behavior, unsafe evidence dir rejection, and failure exit-code preservation.
- Modify `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`: assert host-gated upload covers evidence and policy mentions structured evidence.
- Modify `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`: document structured evidence files under the runtime artifact upload.
- Modify `backlog/tasks/task-2368 - Add-default-VZ-host-smoke-evidence-bundle.md`: keep task status and verification current.

## Task 1: Add Tests For Wrapper Evidence Contract

**Files:**
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`

- [x] **Step 1: Add dry-run default evidence test**

Add a test that runs `run-host-e2e-smoke.sh --dry-run` with a fake bundle/helper and asserts stdout mentions:

```text
evidence directory:
host-smoke-evidence.json
source-bundle-hashes-before.txt
source-bundle-hashes-after.txt
run-bundle-hashes.txt
runtime-paths.txt
cleanup-status.txt
```

Also assert the default evidence path is under `/evidence` beside the default socket runtime dir and that no evidence directory is created.

- [x] **Step 2: Add evidence-dir override dry-run test**

Run dry-run with `--evidence-dir <tmp>/custom-evidence`; assert stdout mentions the override and does not create it.

- [x] **Step 3: Add real fake-helper evidence creation test**

Use the existing fake Python/fake helper pattern. Run a real wrapper execution with socket wait skipped. Assert these files exist in the evidence dir:

```text
host-smoke-evidence.json
source-bundle-hashes-before.txt
source-bundle-hashes-after.txt
run-bundle-hashes.txt
runtime-paths.txt
cleanup-status.txt
```

Parse `host-smoke-evidence.json` and assert it records schema version, source bundle, run bundle, evidence dir, serial dir, helper pid file, `real_host_smoke` success, and cleanup socket absence.

- [x] **Step 4: Add unsafe evidence directory rejection test**

Create an evidence directory with mode `0755`; run the wrapper with `--evidence-dir`; assert non-zero and stderr includes `evidence directory must be owner-only`.

- [x] **Step 5: Add late failure exit-code preservation test**

Use fake Python that succeeds for helper daemon smoke but fails for `test_vz_linux_real_host_e2e.py`. Run the wrapper with `--evidence-dir`. Assert the wrapper exits with the fake pytest failure code and still writes `cleanup-status.txt` plus `host-smoke-evidence.json`.

- [x] **Step 6: Run tests and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
```

Expected: new tests fail because `--evidence-dir` and evidence files do not exist yet.

## Task 2: Implement Evidence Capture In Wrapper

**Files:**
- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`

- [x] **Step 1: Add CLI state and help**

Add:

```bash
EVIDENCE_DIR=""
EVIDENCE_FINALIZED=0
declare -a PHASE_RECORDS=()
```

Add `--evidence-dir PATH` to usage and option parsing.

- [x] **Step 2: Resolve default evidence directory**

After default `IMAGE_STORE_ROOT` resolution, set:

```bash
if [[ -z "${EVIDENCE_DIR}" ]]; then
  EVIDENCE_DIR="$(dirname "${SOCKET_PATH}")/evidence"
fi
```

- [x] **Step 3: Add evidence directory safety helper**

Implement `prepare_private_evidence_dir()` matching serial directory hardening:
refuse symlink, refuse non-directory, create missing directory `0700`, require current owner and no group/world mode.

Dry-run must not call the mutating helper.

- [x] **Step 4: Add phase tracking helpers**

Add `mark_phase_started`, `mark_phase_ok`, and `mark_phase_failed` helpers that append simple `phase|status|timestamp` records. Keep the format shell-safe and easy to serialize.

- [x] **Step 5: Add hash and metadata helpers**

Add helpers for:

```bash
hash_bundle_files SOURCE OUTPUT
write_runtime_paths
write_cleanup_status FINAL_EXIT
write_json_evidence FINAL_EXIT
```

Use a standard-library Python interpreter for SHA-256 calculation and JSON
serialization so paths are quoted safely and the wrapper does not depend on
platform-specific `stat` or `shasum` behavior for evidence content. Do not use
the fake pytest runner shim for evidence serialization in wrapper tests. Store
log paths, sizes, and hashes only; never write raw log contents into JSON.

- [x] **Step 6: Replace simple trap with exit-code-preserving finalize**

Implement:

```bash
finalize() {
  local status="$?"
  set +e
  cleanup
  finalize_evidence "${status}"
  local finalize_status="$?"
  if [[ "${status}" -eq 0 && "${finalize_status}" -ne 0 ]]; then
    exit "${finalize_status}"
  fi
  exit "${status}"
}
trap finalize EXIT INT TERM
```

Guard against double-finalization.

- [x] **Step 7: Wrap phases with explicit status recording**

Call each phase through a small runner helper so failures are recorded before `set -e` exits.

- [x] **Step 8: Run wrapper tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
```

Expected: all wrapper tests pass.

## Task 3: Update Host-Gated Contract And Policy Docs

**Files:**
- Modify: `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`

- [x] **Step 1: Add workflow contract expectation**

Update the workflow contract test to assert the upload path still covers:

```text
${{ runner.temp }}/tldw-vz-helper-ci/**
```

and the policy text includes `host-smoke-evidence.json`.

- [x] **Step 2: Verify RED if policy is not updated**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: fails until policy doc mentions structured evidence files.

- [x] **Step 3: Update policy docs**

Add a short paragraph to `Artifact And Log Expectations` stating that the runtime upload includes an `evidence/` directory containing:

```text
host-smoke-evidence.json
source-bundle-hashes-before.txt
source-bundle-hashes-after.txt
run-bundle-hashes.txt
runtime-paths.txt
cleanup-status.txt
```

- [x] **Step 4: Run contract tests and verify GREEN**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: pass.

## Task 4: Final Verification And Commit

**Files:**
- Modify: `backlog/tasks/task-2368 - Add-default-VZ-host-smoke-evidence-bundle.md`

- [x] **Step 1: Run focused test suite**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: pass.

- [x] **Step 2: Run Bandit on touched executable scope**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r tools/vz-linux-image/scripts -f json -o /tmp/bandit_vz_host_smoke_evidence.json
```

Expected: no new findings in touched script scope.

- [x] **Step 3: Run whitespace validation**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 4: Update Backlog task**

Mark acceptance criteria complete, record verification, document any skipped real VZ smoke, and add final summary.

- [x] **Step 5: Commit**

```bash
git add tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py \
  tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py \
  Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md \
  'backlog/tasks/task-2368 - Add-default-VZ-host-smoke-evidence-bundle.md'
git commit -m "feat: emit VZ host smoke evidence bundle"
```
