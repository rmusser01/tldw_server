# VZ Host-Gated Evidence Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the host-gated VZ Linux workflow upload structured smoke evidence as a separate artifact while narrowing raw helper-log uploads to log-only paths.

**Architecture:** Keep the existing smoke wrapper as the single operator entrypoint. The workflow passes an explicit `--evidence-dir` under its private runtime directory, uploads that path as a dedicated evidence artifact, and keeps a separate helper-log artifact limited to serial/helper logs. Docs and workflow contract tests define the operator artifact order and prevent broad runtime-tree uploads from returning.

**Tech Stack:** GitHub Actions YAML, Bash smoke wrapper contract, pytest workflow/doc contract tests, Markdown operator docs.

---

### Task 1: Add Failing Workflow And Docs Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`
- Read: `.github/workflows/vz-linux-host-gated.yml`
- Read: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`

- [ ] **Step 1: Write failing workflow assertions**

Add coverage that expects:

- `Run managed host smoke` defines `evidence_dir="${runtime_dir}/evidence"`.
- The smoke script args include `--evidence-dir "${evidence_dir}"`.
- There are two upload-artifact steps: `vz-linux-host-gated-evidence` and `vz-linux-host-gated-helper-logs`.
- The evidence artifact path is `${{ runner.temp }}/tldw-vz-helper-ci/evidence/**`.
- The helper-log artifact path does not include the broad `${{ runner.temp }}/tldw-vz-helper-ci/**` runtime-tree glob.

- [ ] **Step 2: Write failing docs assertions**

Add coverage that expects the policy to mention:

- `vz-linux-host-gated-evidence`
- `vz-linux-host-gated-helper-logs`
- evidence as the first artifact to inspect
- helper logs as fallback raw logs
- no disposable image-store/rootfs clone upload through helper logs

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: FAIL because the workflow still has one broad runtime-tree upload and does not pass `--evidence-dir`.

### Task 2: Implement Workflow And Policy Changes

**Files:**
- Modify: `.github/workflows/vz-linux-host-gated.yml`
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`

- [ ] **Step 1: Update the workflow**

In the `Run managed host smoke` step:

- define `evidence_dir="${runtime_dir}/evidence"`
- pass `--evidence-dir "${evidence_dir}"` in the smoke args

In upload steps:

- add a dedicated `Upload smoke evidence` artifact using the existing pinned `actions/upload-artifact` SHA
- keep `Upload helper logs`, but narrow its path to `${{ runner.temp }}/tldw-vz-helper-ci/serial/**`
- keep `if: always()` and `if-no-files-found: ignore` on both upload steps

- [ ] **Step 2: Update docs**

Update host-gated policy and operator notes so contributors know to inspect `vz-linux-host-gated-evidence` first and raw helper logs second. Explicitly state helper logs should not upload disposable image-store/rootfs clones.

- [ ] **Step 3: Run focused GREEN verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
git diff --check
```

Expected: all tests pass, shell syntax passes, diff check is clean.

### Task 3: Finalize

**Files:**
- Modify: `backlog/tasks/task-2332 - Upload-VZ-host-gated-smoke-evidence-artifact.md`

- [ ] **Step 1: Run security check if Python production files changed**

Expected for this slice: Bandit is not required if only YAML, Markdown, and tests change. If production Python changes unexpectedly, run Bandit on the touched Python scope.

- [ ] **Step 2: Update Backlog task**

Record implementation summary, verification commands, known skips, and mark acceptance criteria complete.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/vz-linux-host-gated.yml Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py Docs/superpowers/plans/2026-06-17-vz-host-gated-evidence-artifact-plan.md "backlog/tasks/task-2332 - Upload-VZ-host-gated-smoke-evidence-artifact.md"
git commit -m "ci: upload VZ host smoke evidence artifact"
```
