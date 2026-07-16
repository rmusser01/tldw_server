# PR 2741 Final Review, Rebase, and Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every current PR 2741 review finding, preserve compatibility while centralizing PR-added discovery exceptions, rebase onto current `origin/dev`, and merge after all repository gates are satisfied.

**Architecture:** Keep the change mechanical and compatibility-preserving. Exception implementations move to `app/core/exceptions.py`, while discovery modules import or alias the same class objects so existing public imports remain valid. Logging hardening becomes an effective INFO floor, and cancellation teardown keeps the original exception authoritative.

**Tech Stack:** Python 3.10+, pytest, FastAPI application logging, Git/GitHub CLI, Backlog.md.

---

### Task 1: Centralize PR-added discovery exceptions

**Files:**
- Modify: `tldw_Server_API/tests/Research/test_research_discovery_contracts.py`
- Modify: `tldw_Server_API/app/core/exceptions.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/gateway.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/gateway_adapters.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/executor.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/planner.py`

- [x] **Step 1: Add a failing centralization and compatibility regression**

  Assert that `DiscoveryGatewayError`, `DiscoveryExecutionError`, `DiscoveryAdapterError`, and `PlanningError` are defined in `app.core.exceptions` and are the identical objects re-exported by their existing discovery modules. Assert that the three parser sentinels are imported aliases rather than locally declared exception subclasses.

- [x] **Step 2: Run the regression and verify RED**

  Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_contracts.py -k centralized -vv`

  Expected: FAIL because the seven exception classes are still defined in discovery modules and the core exception exports do not exist.

- [x] **Step 3: Move the seven class definitions with no behavior drift**

  Move the public gateway, executor, adapter, and planner error classes plus the three private parser sentinels into `app/core/exceptions.py`. Keep gateway codes/messages, adapter allowlisting, retry-after validation, and mutation seals intact. Import/re-export the same class identities from the original modules; alias the centralized parser sentinels to the existing private names.

- [x] **Step 4: Run focused tests and verify GREEN**

  Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_contracts.py tldw_Server_API/tests/Research/test_research_discovery_gateway.py tldw_Server_API/tests/Research/test_research_discovery_executor.py tldw_Server_API/tests/Research/test_research_discovery_planner.py -q`

  Expected: PASS with existing exception behavior and import compatibility preserved.

- [x] **Step 5: Commit the exception remediation**

  Commit: `fix(research): centralize discovery exceptions`

### Task 2: Preserve logging policy and explain cancellation suppression

**Files:**
- Modify: `tldw_Server_API/tests/Security/test_http_hop_transport.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/executor.py`

- [x] **Step 1: Add a failing stricter-logging regression**

  Add coverage proving `harden_httpcore_logging()` does not lower explicit or effective `WARNING`/`ERROR` thresholds. Retain existing coverage proving an effective DEBUG threshold is raised to INFO and wire secrets remain suppressed.

- [x] **Step 2: Run the regression and verify RED**

  Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Security/test_http_hop_transport.py -k 'log_hardening or log_levels' -vv`

  Expected: FAIL because the helper currently overwrites stricter logger levels with INFO.

- [x] **Step 3: Implement the minimum effective-level floor**

  For each HTTPcore logger, call `setLevel(logging.INFO)` only when `getEffectiveLevel() < logging.INFO`. Do not add configuration, dependencies, or per-request logging mutation.

- [x] **Step 4: Clarify both symmetric teardown branches**

  Replace each bare cancellation/BaseException validation `pass` with a concise inline explanation that journal validation is best effort and must not mask the original `BaseException`. Do not log or change exception flow.

- [x] **Step 5: Run focused tests and verify GREEN**

  Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Security/test_http_hop_transport.py tldw_Server_API/tests/Research/test_research_discovery_executor.py -q`

  Expected: PASS.

- [x] **Step 6: Commit the logging and teardown remediation**

  Commit: `fix(research): preserve logging and cancellation policy`

### Task 3: Rebase, verify, publish, and close review

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-15-pr-2741-final-review-rebase-merge-plan.md`
- Modify: `backlog/tasks/task-12968.9 - Finish-PR-2741-review-rebase-and-merge.md`

- [ ] **Step 1: Rebase onto the actual latest dev**

  Fetch `origin/dev`, record the old head and merge base, then rebase the branch. Use `git range-diff` to confirm intended patch equivalence and resolve only genuine conflicts.

- [ ] **Step 2: Run fresh integrated verification**

  Run the complete Research suite; focused Security suite; inventory Node/Python validators and authoritative contract; Ruff and Black on touched non-baseline files; Python 3.10 compilation; JSON/Node syntax and `git diff --check`; Bandit on touched Python scope.

- [ ] **Step 3: Obtain independent final review**

  Dispatch a correctness/spec reviewer and a code-quality/minimality reviewer against `origin/dev...HEAD`. Fix every valid Critical or Important finding and re-run affected gates.

- [ ] **Step 4: Publish safely and close GitHub review**

  Push the rebased branch with `--force-with-lease`. Reply to roots `3592094302`, `3592094308`, and `3592094311` with exact fixes/evidence, resolve their threads, and re-query for new roots or unresolved threads.

- [ ] **Step 5: Satisfy the human and CI merge gates**

  Keep the PR draft until the requester supplies a human-authored Change summary. Update the PR body with that text without rewriting it. Mark ready only after the summary is present; wait for required checks and fix any actual GitHub Actions failures test-first.

- [ ] **Step 6: Merge and verify the result**

  Merge PR 2741 using the repository-supported method, confirm the PR state is `MERGED`, and confirm the merge commit is reachable from current `origin/dev`. Finalize `TASK-12968.9` with exact verification, review, CI, and merge evidence.
