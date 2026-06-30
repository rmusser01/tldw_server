# ACP Stage 1 Retention Redaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the ACP Stage 1 release-hardening gap by enforcing retention cleanup and adding redacted session drill-through views.

**Architecture:** Keep the change ACP-local. Add retention enforcement to the existing ACP session store cleanup path and reuse one endpoint-local redaction helper for detail, events, and artifacts so full-fidelity operator views remain unchanged unless callers request redacted output.

**Tech Stack:** FastAPI, Pydantic response models, SQLite-backed ACP session/audit stores, pytest.

---

### Task 1: Retention Maintenance

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/config.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ACP_Audit_DB.py`
- Modify: `tldw_Server_API/app/services/admin_acp_sessions_service.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_management.py`

- [x] Write failing tests for closed-session hard-delete retention and app-path audit/session cleanup.
- [x] Run focused tests and confirm they fail for missing retention APIs.
- [x] Add `ACP_SESSION_RETENTION_DAYS` config and store-level retention configuration.
- [x] Add closed/error session purge that cascades `session_messages`.
- [x] Wire audit purge and session purge through the existing ACP cleanup task path.
- [x] Run focused retention tests and confirm they pass.

### Task 2: Redacted ACP Views

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_hardening_controls.py`

- [x] Write failing tests for `?redacted=true` detail, events, and artifacts.
- [x] Run focused tests and confirm they fail for missing redacted behavior.
- [x] Add shared ACP redaction helpers for transcript, event payload, and artifact payload use.
- [x] Add redacted query mode to detail, events, and artifacts without changing default full-fidelity responses.
- [x] Run focused redaction tests and confirm they pass.

### Task 3: Docs And Closeout

**Files:**
- Modify: `Docs/Development/Agent_Client_Protocol.md`
- Modify: `Docs/Development/ACP_Production_Readiness.md`
- Modify: `backlog/tasks/task-241 - ACP-Stage-1-retention-and-redaction-implementation.md`

- [x] Update operator docs to describe retention and redacted-view policy.
- [x] Update readiness caveats to remove stale blocked language if verification passes.
- [x] Run focused ACP tests, Bandit on touched Python paths, and `git diff --check`.
- [x] Update TASK-241 with verification and final summary.
