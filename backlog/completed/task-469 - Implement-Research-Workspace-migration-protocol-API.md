---
id: TASK-469
title: Implement Research Workspace migration protocol API
status: Done
labels:
- research-workspace
- workspaces
- backend
- migration
priority: High
references:
- Docs/superpowers/plans/2026-05-23-research-workspace-migration-protocol-api-plan.md
modified_files:
- Docs/Design/Research_Workspace_Migration_Protocol_API.md
- Docs/superpowers/plans/2026-05-23-research-workspace-migration-protocol-api-plan.md
- tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a durable backend protocol for Research Workspace migration sessions, chunk receipts, recovery manifests, and safe finalize/ack behavior without route aliases or client-storage deletion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migration routes live under /api/v1/workspaces/migrations and do not conflict with dynamic workspace IDs.
- [x] #2 Migration sessions and chunk receipts persist in ChaChaNotes DB with idempotent create/read behavior.
- [x] #3 Chunk validation enforces bounded sizes and hashes and rejects conflicting duplicate chunks.
- [x] #4 Finalize returns a durable receipt/recovery manifest and never authorizes browser-storage deletion in this slice.
- [x] #5 Tests cover happy path, idempotency, route ordering, validation failures, and deletion-ack guard.
- [x] #6 Verification includes pytest, Bandit on touched Python scope, and a real backend/API check.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-research-workspace-migration-protocol-api-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented bounded Pydantic schemas, a dedicated router mounted before dynamic workspaces, ChaChaNotes v47 migration tables, idempotent session/chunk persistence, finalize recovery manifest generation, and the client deletion safety guard. Added a zero-byte declared chunk regression during review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: migration API tests pass (7 passed); broader workspace regression suite passes (46 passed, 6 warnings); git diff --check passed for touched files; Bandit on touched Python scope reported 0 findings; live FastAPI backend validation on 127.0.0.1:18001 returned create/list/get/chunk/finalize/delete-ack statuses 201/200/200/200/200/409. Known follow-up: source ingestion/extraction/chunking/indexing Jobs ownership remains a separate first-class workspace ingestion/status slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
