# Prototype Risk Gate 7 Ops Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #1460 by documenting prototype workspace operational visibility, support workflows, and owner/collaborator examples needed before the Risk Gate 8 release review.

**Architecture:** Keep this as a documentation-first slice. Add one operator runbook, one user-facing lifecycle guide, small links/status-field updates to existing prototype API/contract docs, and a focused pytest guard that checks the required support examples remain present.

**Tech Stack:** Markdown docs under `Docs/`, existing pytest suite under `tldw_Server_API/tests/PrototypeWorkspaces`, Backlog.md task tracking.

---

### Task 1: Operator Runbook

**Files:**
- Create: `Docs/Operations/Prototype_Workspaces_Runbook.md`

- [x] **Step 1: Draft the runbook sections**

Include:
- setup/configuration prerequisites
- signing secret expectations and rotation posture
- runtime bootstrap and Jobs behavior
- preview health and preview grant diagnosis
- quotas/rate limits and current gaps
- promotion validation/promotion request diagnosis
- status fields and audit/support breadcrumbs
- incident triage checklist

- [x] **Step 2: Verify runbook references current contract terms**

Run: `rg -n "runtime_status|preview_status|preview_health|promotion_requests|retryable|signing secret" Docs/Operations/Prototype_Workspaces_Runbook.md`

Expected: every term appears in the relevant operational section.

### Task 2: User Lifecycle Guide

**Files:**
- Create: `Docs/User_Guides/Prototype_Workspaces.md`
- Modify: `Docs/User_Guides/index.md`

- [x] **Step 1: Draft owner and collaborator flows**

Include:
- owner creates workspace and share link
- collaborator enters password-protected link
- single-use/exhausted link behavior
- resume cookie behavior
- revoked/expired/archived link outcomes
- collaborator branch/session expectations
- promotion request, rejection, validation failure, conflict, and success outcomes

- [x] **Step 2: Verify user examples cover all required issue scenarios**

Run: `rg -n "password-protected|single-use|resume cookie|revoked|archived|exhausted|promotion conflict" Docs/User_Guides/Prototype_Workspaces.md`

Expected: every required scenario appears.

### Task 3: API And Contract Cross-Links

**Files:**
- Modify: `Docs/API-related/Prototype_Workspaces_API.md`
- Modify: `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`

- [x] **Step 1: Add a Risk Gate 7 section to the API guide**

Document where operators/users should look for runbooks, user guides, status fields, job result fields, and support diagnosis.

- [x] **Step 2: Add operational status/support checklist to the contract matrix**

List fields available for support:
- workspace `canonical_preview_status`, `publish_validation_status`
- session `runtime_status`, `preview_status`, `last_saved_snapshot_id`, `expires_at`, `revoked_at`
- snapshot `preview_health`
- promotion request `status`, `reviewed_by_user_id`, `review_notes`
- job response `job_id`, `job_type`, `status`, `idempotency_key`
- structured error `category`, `frontend_state`, `retryable`

### Task 4: Docs Contract Guard

**Files:**
- Create: `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py`

- [x] **Step 1: Add a focused docs presence test**

Use `pathlib.Path` to read the new runbook and user guide. Assert they include the required operational field names and issue-required examples.

- [x] **Step 2: Run the docs guard**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py -q`

Expected: pass.

### Task 5: Verification And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-481 - Risk-Gate-7-prototype-operational-visibility-and-documentation.md`

- [x] **Step 1: Run focused verification**

Run:
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py -q`
- `git diff --check`
- Bandit is not required unless production Python code changes; document the skip if docs/test-only.

- [x] **Step 2: Update Backlog task**

Mark acceptance criteria and DoD complete, record verification and known skips, and add final summary.

- [x] **Step 3: Commit**

Run:
- `git add Docs/Operations/Prototype_Workspaces_Runbook.md Docs/User_Guides/Prototype_Workspaces.md Docs/API-related/Prototype_Workspaces_API.md Docs/API-related/Prototype_Workspaces_Contract_Matrix.md tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_docs_contract.py "backlog/tasks/task-481 - Risk-Gate-7-prototype-operational-visibility-and-documentation.md" Docs/superpowers/plans/2026-05-22-prototype-risk-gate-7-ops-docs-plan.md`
- `git commit -m "Document prototype workspace operational visibility"`
