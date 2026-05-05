# Sandbox Runtime Status Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize known sandbox runtime status messages into stable reason codes while preserving raw runner diagnostics.

**Architecture:** Keep the taxonomy centralized in `run_status_taxonomy.py`. Add explicit alias sets for known policy and runtime-unavailable messages, preserve conservative heuristics for existing compatibility, and document the remaining Phase 3 scope.

**Tech Stack:** Python 3.11, pytest, existing sandbox status schemas and runtime taxonomy helpers.

---

### Task 1: Add Failing Taxonomy Tests

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py`

- [x] **Step 1: Add regression tests for policy aliases**

Add a test that loops over:

```python
[
    "lima_policy_failed",
    "vz_linux_policy_failed",
    "vz_macos_policy_failed",
    "seatbelt_policy_failed",
    "worktree_policy_failed",
]
```

and expects `normalize_run_status_reason(...) == "policy_failed"`.

- [x] **Step 2: Add runtime-unavailable alias tests**

Add a test covering exact unavailable aliases such as `docker_unavailable`,
`firecracker_unavailable`, `vz_linux_unavailable`, `vz_macos_unavailable`,
`seatbelt_unavailable`, and `worktree_unavailable`.

- [x] **Step 3: Verify red**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q
```

Expected: failure showing VZ policy aliases still normalize as `runtime_unavailable`.

### Task 2: Implement Central Alias Sets

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py`

- [x] **Step 1: Add `_POLICY_FAILED_MESSAGES`**

Move known policy failure messages into a dedicated exact alias set.

- [x] **Step 2: Remove policy failures from `_RUNTIME_UNAVAILABLE_MESSAGES`**

Keep runtime-unavailable aliases limited to runtime availability and
provisioning failures.

- [x] **Step 3: Check policy aliases before runtime unavailable**

In the failed-phase branch, return `policy_failed` for exact policy aliases and
the existing conservative policy substring fallback before checking runtime
unavailable.

- [x] **Step 4: Verify green**

Run the focused taxonomy tests again.

### Task 3: Update Docs And Task Tracking

**Files:**
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `backlog/tasks/task-57 - Normalize-sandbox-runtime-status-reason-taxonomy.md`

- [x] **Step 1: Update current gaps**

Change the Phase 3 gap text to say the first alias pass is complete and richer
structured error metadata remains future work.

- [x] **Step 2: Update Backlog task**

Check completed acceptance criteria and record verification.

### Task 4: Verification And Commit

**Files:**
- Verify touched Python and docs.

- [x] **Step 1: Run focused tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q
```

- [x] **Step 2: Run Bandit on touched Python**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py -f json -o /tmp/bandit_sandbox_runtime_taxonomy.json
```

- [x] **Step 3: Run whitespace check**

```bash
git diff --check
```

- [x] **Step 4: Commit**

```bash
git add Docs/Sandbox/sandbox-runtime-capability-inventory.md Docs/superpowers/specs/2026-05-05-sandbox-runtime-status-taxonomy-design.md Docs/superpowers/plans/2026-05-05-sandbox-runtime-status-taxonomy-implementation-plan.md "backlog/tasks/task-57 - Normalize-sandbox-runtime-status-reason-taxonomy.md" tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py
git commit -m "fix(sandbox): normalize runtime status reason aliases"
```
