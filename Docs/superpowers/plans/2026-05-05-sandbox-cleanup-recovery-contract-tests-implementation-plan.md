# Sandbox Cleanup And Recovery Contract Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local-first sandbox cleanup/recovery contract coverage without requiring host-gated runtimes.

**Architecture:** Keep the slice test-first and portable. Use runner unit seams for process/worktree cleanup, service/orchestrator seams for durable session cleanup, and runtime capability metadata for no-warm-reuse host-local contracts. Avoid production edits unless a failing test exposes a real behavior gap.

**Tech Stack:** Python 3.11, pytest, existing sandbox service/orchestrator/store helpers, git worktree runner fakes.

---

### Task 1: Worktree Timeout Cleanup Contract

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_worktree_runner.py`

- [x] **Step 1: Write the timeout cleanup contract test**

Add a test that fakes `WorktreeRunner.create_worktree`, `destroy_worktree`,
`tempfile.mkdtemp`, and `subprocess.Popen`. The fake process raises
`subprocess.TimeoutExpired` from `wait()`. Assert the resulting status is
`RunPhase.timed_out`, the created worktree is destroyed, the run directory is
removed, and `_active_proc` / `_active_run_dir` do not retain the run id.

- [x] **Step 2: Run the focused test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_worktree_runner.py::test_start_run_timeout_cleans_worktree_run_dir_and_active_tracking -q
```

Expected: fail if timeout cleanup is incomplete, or pass immediately if the
current implementation already satisfies the contract.

- [x] **Step 3: Implement only if needed**

If the test fails because cleanup is incomplete, update
`tldw_Server_API/app/core/Sandbox/runners/worktree_runner.py` to keep cleanup in
`finally`: destroy the worktree, clear active tracking, remove the run directory,
and preserve the `execution_timeout` status.

Outcome: no production change was needed; the contract test passed against the
current implementation.

- [x] **Step 4: Re-run worktree tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_worktree_runner.py -q
```

Expected: pass.

### Task 2: Durable Session Cleanup And Stale Metadata Contracts

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_session_store_durability.py`

- [x] **Step 1: Write the durable deletion workspace cleanup test**

Add a test using the SQLite sandbox store helper. Create a session in one
service, write a marker under its workspace, instantiate a second service, and
destroy the session through the second service. Assert the persisted session is
gone and the original session root is removed.

- [x] **Step 2: Write or preserve stale metadata rejection coverage**

Confirm the existing stale cached session-backed run test rejects store lookup
failure without enqueueing. Add only a narrow assertion if the contract is not
explicit enough.

Outcome: existing stale-cache rejection coverage already asserts the no-enqueue
contract; this slice added the missing cross-service durable destroy cleanup
coverage.

- [x] **Step 3: Run focused session durability tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_session_store_durability.py -q
```

Expected: pass, or expose a real cleanup/recovery gap to fix in
`SandboxService`/`SandboxOrchestrator`.

### Task 3: Host-Local No Warm Reuse Contract

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`

- [x] **Step 1: Add explicit host-local session contract test**

Assert `seatbelt` and `worktree` session metadata both report
`reuse_model="workspace_only"`, `requires_live_health_check=False`,
`recovery_state="unsupported"`, and `repair_state="unsupported"`.

- [x] **Step 2: Run runtime inventory tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q
```

Expected: pass.

### Task 4: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-62 - Add-sandbox-cleanup-and-recovery-contract-tests.md`

- [x] **Step 1: Run all focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_worktree_runner.py tldw_Server_API/tests/sandbox/test_session_store_durability.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q
```

Expected: pass.

- [x] **Step 2: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 3: Run Bandit if production code changed**

If only docs/tests/backlog changed, document the skip in TASK-62. If production
code changed, run Bandit on touched production paths.

Outcome: skipped; no production code changed.

- [x] **Step 4: Update TASK-62**

Check completed acceptance criteria and Definition of Done items, append
verification notes, and add a final summary.

- [x] **Step 5: Commit**

Run:

```bash
git add Docs/superpowers/specs/2026-05-05-sandbox-cleanup-recovery-contract-tests-design.md Docs/superpowers/plans/2026-05-05-sandbox-cleanup-recovery-contract-tests-implementation-plan.md tldw_Server_API/tests/sandbox/test_worktree_runner.py tldw_Server_API/tests/sandbox/test_session_store_durability.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py "backlog/tasks/task-62 - Add-sandbox-cleanup-and-recovery-contract-tests.md"
git commit -m "test(sandbox): add cleanup recovery contract coverage"
```
