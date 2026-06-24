# Agent Orchestration Review Hardening Implementation Plan

> Backlog task: TASK-2403

## Stage 1: Retry Dispatch State Gate
**Goal**: Allow reviewer-rejected tasks to run again without permitting duplicate active runs.
**Success Criteria**: A task returned to `inprogress` by a rejected review can dispatch a new run after prior runs are terminal; a task with an active running run rejects duplicate dispatch.
**Tests**: Focused API dispatch tests in `tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py` and DB unit coverage for active-run rejection.
**Status**: Complete

## Stage 2: Per-User DB Scoping
**Goal**: Make `OrchestrationDB` project, task, run, review, and MCP-server methods enforce the instance user boundary.
**Success Criteria**: Shared-database tests prove user 2 cannot read, list, transition, review, or mutate user 1 records.
**Tests**: Cross-user tests in `tldw_Server_API/tests/Agent_Orchestration/test_orchestration_db.py` and `test_workspace_db.py`.
**Status**: Complete

## Stage 3: Run State Machine
**Goal**: Reject missing run updates and terminal run rewrites.
**Success Criteria**: Completing or failing an already terminal run raises a deterministic transition error; missing run updates raise `OrchestrationNotFoundError`.
**Tests**: Run lifecycle tests in `test_orchestration_db.py` plus model transition helper tests.
**Status**: Complete

## Stage 4: Artifact Bounds
**Goal**: Add explicit limits for ACP completion artifact count and promoted artifact payload size.
**Success Criteria**: Oversized artifact lists fail completion validation; oversized promotable payloads are skipped with structured error reasons before DB writes.
**Tests**: Parser and promotion tests in `test_orchestration_api.py` and `test_artifact_promotion.py`.
**Status**: Complete

## Stage 5: Legacy Service Isolation
**Goal**: Keep production imports pointed at a SQLite DB factory and remove stale architecture guidance around the legacy in-memory service.
**Success Criteria**: Production routes and resolver import the factory from a dedicated module; README and tests reflect the SQLite-backed path as the production contract.
**Tests**: Existing route/resolver tests continue to patch the route-level factory and a focused factory test covers `OrchestrationDB.for_user`.
**Status**: Complete

## Verification

- `TEST_MODE=1 ULTRA_MINIMAL_APP=1 python -m pytest --confcutdir=tldw_Server_API/tests/Agent_Orchestration tldw_Server_API/tests/Agent_Orchestration -q`
  - Result: 204 passed, 2 warnings.
- `python -m bandit -r tldw_Server_API/app/core/Agent_Orchestration tldw_Server_API/app/core/DB_Management/Orchestration_DB.py tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py tldw_Server_API/app/services/mcp_hub_workspace_root_resolver.py -f json -o /tmp/bandit_agent_orchestration_2403.json`
  - Result: 0 findings, 0 errors.
