# Prototype Risk Gate 1 Auth Invariants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Risk Gate 1 for prototype workspace collaboration by documenting the security model, updating the draft frontend/backend contract, and adding focused backend coverage for authorization invariants that already exist or must be enforced now.

**Architecture:** Keep this slice contract-first and narrow. Documentation records the actor model, non-enumerating states, token/session dispositions, audit/quota deferrals, and frontend prep requirements. Backend code changes are limited to promotion-request authorization so revoked or expired shared actors cannot use still-signed collaborator session tokens.

**Tech Stack:** FastAPI, pytest, SQLite-backed AuthNZ fixtures, Markdown documentation.

---

### Task 1: Record The Risk Gate 1 Security Plan

**Files:**
- Create: `Docs/superpowers/plans/2026-05-13-prototype-risk-gate-1-auth-invariants.md`
- Modify: `backlog/tasks/task-324 - Risk-Gate-1-prototype-workspace-threat-model-and-auth-invariants.md`

- [x] **Step 1: Create this plan in the isolated worktree**

Run: `git status --short --branch`

Expected: clean branch `codex/prototype-risk-gate-1-auth-invariants` tracking `origin/dev` except for this plan.

- [x] **Step 2: Record plan location on TASK-324**

Use Backlog.md MCP `task_edit` to add the plan path and note that baseline `test_service_authorization.py` passed before edits.

Expected: task notes mention the clean worktree and focused baseline.

### Task 2: Add Failing Promotion Authorization Tests

**Files:**
- Modify: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`

- [x] **Step 1: Add revoked shared actor promotion test**

Add a test that creates a workspace, external actor, branch session, and candidate snapshot, revokes the shared actor, then posts to `POST /api/v1/prototype-promotions` with the still-valid session token.

Expected behavior: response `403` with `Prototype session token is no longer active`.

- [x] **Step 2: Add expired shared actor promotion test**

Add a similar test that sets `prototype_shared_actors.expires_at` to a past timestamp before posting the promotion request.

Expected behavior: response `403` with `Prototype session token is no longer active`.

- [x] **Step 3: Run the new tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py \
  -k "revoked_shared_actor_cannot_submit_promotion_request or expired_shared_actor_cannot_submit_promotion_request" -q
```

Expected: both tests fail because the current endpoint accepts the stale signed session token.

### Task 3: Enforce Active External Actor On Promotion Submission

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py`
- Test: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py`

- [x] **Step 1: Add a small datetime parser for endpoint guards**

Add a local helper that parses optional ISO-8601 timestamps and normalizes naive timestamps to UTC.

Expected: helper returns `None` for blank/malformed values and timezone-aware `datetime` for valid values.

- [x] **Step 2: Add `_external_actor_inactive` helper**

Add a helper that treats missing, revoked, or expired shared actors as inactive.

Expected: it checks both `is_revoked` and `revoked_at`, plus `expires_at <= now`.

- [x] **Step 3: Guard promotion request submission**

In `create_promotion_request`, after the branch session matches the token actor, fetch the shared actor and reject if it is missing, revoked, expired, bound to another workspace, or bound to another share link.

Expected: inactive or mismatched actor returns `403` with `Prototype session token is no longer active`.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the same focused pytest command from Task 2.

Expected: both tests pass.

### Task 4: Write Threat Model And Contract Draft

**Files:**
- Create: `Docs/Security/Prototype_Workspaces_Threat_Model.md`
- Modify: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`
- Modify: `Docs/API-related/Prototype_Workspaces_API.md`

- [x] **Step 1: Create the threat model**

Document actor identities, trust boundaries, authorization invariants, non-enumerating error policy, token/session requirements with dispositions, audit events, quota policy, deferred items, and verification evidence.

Expected: every Risk Gate 1 requirement is explicitly covered.

- [x] **Step 2: Update the contract matrix**

Replace TBD token/session dispositions with concrete dispositions and add frontend fixture/mock-state prep notes plus route-state audit checklist.

Expected: no `TBD` remains in the Risk Gate 1 sections.

- [x] **Step 3: Link the threat model from the API doc**

Add a short security-contract section to `Prototype_Workspaces_API.md`.

Expected: API docs point maintainers to the Risk Gate 1 source of truth.

### Task 5: Verify And Close Out

**Files:**
- Modify: `backlog/tasks/task-324 - Risk-Gate-1-prototype-workspace-threat-model-and-auth-invariants.md`

- [x] **Step 1: Run backend prototype tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/PrototypeWorkspaces/test_service_authorization.py \
  tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_link_exchange.py \
  tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_endpoints.py -q
```

Expected: relevant prototype authorization tests pass.

- [x] **Step 2: Run Bandit on touched backend code**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py \
  -f json -o /tmp/bandit_prototype_risk_gate_1.json
```

Expected: no new actionable findings in the touched endpoint.

- [x] **Step 3: Run whitespace verification**

Run: `git diff --check`

Expected: no whitespace errors.

- [x] **Step 4: Update TASK-324**

Check completed acceptance criteria, record verification commands/results, and add final summary.

Expected: Backlog task has enough handoff context for review.
