# VZ Linux Host Failure Drills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manual opt-in VZ Linux host failure drills that verify stale same-session VM recovery on prepared Apple Silicon hosts.

**Architecture:** Reuse the existing host smoke script as the single helper lifecycle entrypoint. Keep default smoke stable by running only the existing host-smoke marker unless `--include-failure-drills` is passed. The drill itself lives in the real-host pytest module and invalidates only a VM created by the current test session.

**Tech Stack:** Bash, GitHub Actions, pytest, FastAPI sandbox service internals, macOS Virtualization helper client.

---

## Preconditions

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/sandbox-host-failure-drills`
- Branch: `codex/sandbox-host-failure-drills`
- Backlog: `TASK-145`
- Spec: `Docs/superpowers/specs/2026-05-09-vz-linux-host-failure-drills-design.md`
- Base branch: `origin/dev`

## File Map

- Modify: `.github/workflows/vz-linux-host-gated.yml`
  - Add manual `include_failure_drills` input and conditionally pass
    `--include-failure-drills` to the smoke script.
- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
  - Add script flag, usage text, dry-run behavior, and conditional pytest marker
    invocation.
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`
  - Add script dry-run/help coverage for the new flag.
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
  - Add the real-host stale session VM recovery drill behind a dedicated marker.
- Modify: `pyproject.toml`
  - Register the new strict pytest marker.
- Modify: `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`
  - Add workflow contract coverage for the new manual input and conditional flag
    wiring.
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
  - Document manual-only failure drills and default scheduled-run behavior.
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
  - Briefly mention the opt-in failure-drill coverage.
- Modify: `backlog/tasks/task-145 - Add-manual-opt-in-VZ-Linux-host-failure-drills.md`
  - Keep status, verification, and final notes current.

## Task 1: Script Flag And Focused Tests

**Files:**
- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`

- [ ] **Step 1: Add failing tests for default and opt-in dry-run behavior**

Add assertions that `--help` includes `--include-failure-drills`, that default
dry-run output does not include `vz_linux_host_failure_drill`, and that dry-run
with `--include-failure-drills` does include it.

- [ ] **Step 2: Run focused script tests and confirm failure**

Run:

```bash
python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
```

Expected: fail because the flag is not implemented.

- [ ] **Step 3: Implement the script flag**

Add `INCLUDE_FAILURE_DRILLS=0`, parse `--include-failure-drills`, document it in
usage, add `run_real_vz_linux_failure_drills`, and conditionally call it after
`run_real_vz_linux_host_smoke`.

- [ ] **Step 4: Re-run focused script tests**

Run:

```bash
python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
```

Expected: pass.

## Task 2: Workflow Input And Contract Tests

**Files:**
- Modify: `.github/workflows/vz-linux-host-gated.yml`
- Modify: `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`

- [ ] **Step 1: Add failing workflow contract assertions**

Assert the workflow defines `workflow_dispatch.inputs.include_failure_drills`,
that it defaults to false, and that the managed smoke step conditionally appends
`--include-failure-drills`.

- [ ] **Step 2: Run the workflow contract test and confirm failure**

Run:

```bash
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: fail because the input and wiring do not exist.

- [ ] **Step 3: Add workflow input and conditional script argument**

Add a boolean manual input and append `--include-failure-drills` only for truthy
manual input values. Do not enable it for scheduled runs by default and do not
change branch gating or triggers.

- [ ] **Step 4: Re-run the workflow contract test**

Run:

```bash
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: pass.

## Task 3: Real-Host Failure Drill

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the failing real-host drill test**

Add `@pytest.mark.vz_linux_host_failure_drill` to a test that:

- creates a real `vz_linux` session
- runs a first command
- reads the session-control VM ID
- terminates that VM through `VZLinuxRunner.helper_client_cls()`
- runs a second command in the same session
- asserts completion and a changed VM ID
- destroys the session in `finally`

- [ ] **Step 2: Register the pytest marker**

Add `vz_linux_host_failure_drill` to the `pyproject.toml` pytest marker list so
strict marker collection works on every host.

- [ ] **Step 3: Run marker selection in host-independent mode**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -m vz_linux_host_failure_drill -q -rs
```

Expected on non-prepared hosts: skipped with the existing opt-in/host reasons.

- [ ] **Step 4: Ensure the drill uses existing fallback behavior**

No production code change should be needed if `VZLinuxRunner` already treats
failed or absent VM status probes as not healthy and provisions a fresh VM.
Only adjust production code if the test exposes a still-valid regression.

## Task 4: Documentation

**Files:**
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

- [ ] **Step 1: Document the manual-only failure drill contract**

Update the host-gated policy to explain that failure drills are available only
through manual dispatch or explicit script flag, are disabled for scheduled runs
by default, and must stay scoped to resources created by the drill itself.

- [ ] **Step 2: Add a short sandbox README note**

Mention `--include-failure-drills` near the existing host smoke guidance.

## Task 5: Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-145 - Add-manual-opt-in-VZ-Linux-host-failure-drills.md`

- [ ] **Step 1: Run focused verification**

Run:

```bash
python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -m vz_linux_host_failure_drill -q -rs
python -m bandit -r tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -s B101 -f json -o /tmp/bandit_vz_host_failure_drills.json
```

Expected: focused tests pass or host-gated drill test skips on non-prepared
hosts; Bandit produces no new findings outside skipped test assert noise.

- [ ] **Step 2: Update Backlog task**

Check completed acceptance criteria, record verification results, add final
summary, and document any real-host skip reason if the local host is not
prepared for VZ execution.

- [ ] **Step 3: Self-review changed files**

Review the diff for:

- failure drills disabled by default
- no schedule/nightly enablement
- no PR/push trigger changes
- no broad VM cleanup
- clear docs and operator wording

- [ ] **Step 4: Commit**

Commit all changes with a concise message such as:

```bash
git add .github/workflows/vz-linux-host-gated.yml \
  tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  tools/vz-linux-image/tests/test_host_e2e_smoke_script.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py \
  Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md \
  tldw_Server_API/app/core/Sandbox/README.md \
  Docs/superpowers/specs/2026-05-09-vz-linux-host-failure-drills-design.md \
  Docs/superpowers/plans/2026-05-09-vz-linux-host-failure-drills-implementation-plan.md \
  "backlog/tasks/task-145 - Add-manual-opt-in-VZ-Linux-host-failure-drills.md"
git commit -m "test(sandbox): add manual VZ host failure drills"
```
