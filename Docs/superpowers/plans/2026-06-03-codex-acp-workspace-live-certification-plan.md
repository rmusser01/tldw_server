# Codex ACP Workspace Live Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in certification path that can validate a Research Workspace-launched Codex ACP session for workspace linkage, MCP context evidence, artifacts, reviewer-loop diagnostics, sandbox evidence, and bounded live output.

**Architecture:** Extend the existing `acp_certification_smoke.py` helper instead of adding another harness. The new profile should remain live-environment gated, report granular capability states, and preserve the current `live-e2e` behavior.

**Tech Stack:** Python helper script, pytest unit coverage, existing ACP REST endpoints, Backlog.md task tracking, ACP certification docs.

---

### Task 1: Define Workspace Certification Contract

**Files:**
- Modify: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Test: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`
- Update: `Docs/Development/ACP_Certification_Checklist.md`
- Update: `Docs/Development/ACP_Compatibility_Matrix.md`

- [x] **Step 1: Write failing manifest tests**

Add tests proving a `workspace-live-e2e` manifest exists, requires the same live backend env as `live-e2e`, declares `ACP_E2E_WORKSPACE_ID` as optional workspace evidence input, and advertises capability IDs for `workspace_env`, `mcp_injection`, `artifacts`, `review_loop`, `sandbox`, `diagnostics`, and `redacted_support_view`.

- [x] **Step 2: Verify RED**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py::test_workspace_live_e2e_manifest_declares_workspace_capabilities`

Expected: FAIL because the manifest does not exist yet.

- [x] **Step 3: Add minimal manifest**

Add `workspace-live-e2e` to `_MANIFESTS` with one command invoking `--backend-workspace-live-e2e`, keeping `safe_to_run_by_default: false`.

- [x] **Step 4: Verify GREEN**

Run the focused test again and confirm it passes.

### Task 2: Add Backend Workspace Live Runner

**Files:**
- Modify: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Test: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [x] **Step 1: Write failing runner test**

Add a fake-HTTP test for `_run_backend_workspace_live_e2e_from_env()` that asserts the helper:

- calls health and setup guide
- posts `/api/v1/acp/sessions/new` with `workspace_id`, non-empty `mcp_servers`, and a workspace-specific session name
- prompts for an artifact-like response
- queries detail, events, artifacts, diagnostics, and workspace-filtered `/api/v1/acp/sessions?workspace_id=...`
- reports granular evidence states in the PASS JSON output
- closes the session on success and failure

- [x] **Step 2: Verify RED**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py::test_backend_workspace_live_e2e_runs_workspace_evidence_sequence`

Expected: FAIL because the runner function does not exist yet.

- [x] **Step 3: Implement minimal runner**

Implement `_run_backend_workspace_live_e2e_from_env()` using the existing `_http_json_request`, `_check_backend_response`, `_backend_e2e_timeout_seconds`, and close-on-finally pattern. Use bounded JSON output and never print API keys or raw full transcripts.

- [x] **Step 4: Wire CLI flag and manifest run**

Add `--backend-workspace-live-e2e` and route the `workspace-live-e2e` manifest command through it. Preserve `--backend-live-e2e` behavior.

- [x] **Step 5: Verify GREEN**

Run the focused helper tests and then the full helper test file.

### Task 3: Document Evidence Semantics

**Files:**
- Modify: `Docs/Development/ACP_Certification_Checklist.md`
- Modify: `Docs/Development/ACP_Compatibility_Matrix.md`
- Modify: `backlog/tasks/task-509 - Certify-live-Codex-ACP-workspace-launch-flow.md`

- [x] **Step 1: Update checklist**

Document `workspace-live-e2e`, required/optional env vars, skip semantics, and the difference between host live E2E and workspace live certification.

- [x] **Step 2: Update matrix caveat language**

Do not claim Codex workspace certification has passed unless a live run succeeds. Add the new command as the recommended follow-up for the existing Codex row’s skipped workspace/MCP/artifact/reviewer/sandbox gaps.

- [x] **Step 3: Record Backlog implementation notes**

Add the plan path, touched files, and verification commands to `TASK-509`.

### Task 4: Verify and Commit

**Files:**
- All touched files above

- [x] **Step 1: Run focused tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [x] **Step 2: Run Bandit on touched Python**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -q Helper_Scripts/Testing-related/acp_certification_smoke.py`

- [x] **Step 3: Attempt live environment check**

If a backend is already running and required env vars are available, run the new `workspace-live-e2e` helper. If not, record the exact missing prerequisite instead of claiming live certification.

Result: helper refused with exit 2 because `TLDW_E2E_SERVER_URL`, `TLDW_E2E_API_KEY`, and `ACP_AGENT_PROFILE` were not present in the shell. No live certification claim was made.

- [x] **Step 4: Commit**

Stage the helper, tests, docs, Backlog task, and this plan. Commit with message `test: add Codex ACP workspace certification harness`.
