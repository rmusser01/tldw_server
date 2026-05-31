---
id: TASK-575
title: Implement MCP Unified Stage 4L editable profile CRUD
status: In Progress
labels:
- mcp-unified
- implementation
- stage-4l
modified_files:
- mcp_unified/interfaces/storage.py
- mcp_unified/gateway/profiles.py
- mcp_unified/storage/sqlite.py
- mcp_unified/gateway/fastapi.py
- mcp_unified/gateway/cli.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Stage 4L plan for manager-owned editable profile create, limited patch, guarded delete, FastAPI routes, CLI commands, focused tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Manager-owned create, limited patch, and guarded delete are implemented.
- [ ] #2 Persistent-store delete uses an atomic or serialized guarded-delete path.
- [ ] #3 FastAPI exposes `POST /profiles`, `PATCH /profiles/{profile_id}`, and `DELETE /profiles/{profile_id}` with deterministic reason-code mapping.
- [ ] #4 CLI exposes `create-profile`, `patch-profile`, and `delete-profile` with file/stdin JSON input where applicable.
- [ ] #5 Focused manager, FastAPI, CLI, boundary, and Bandit verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation follows `Docs/superpowers/plans/2026-05-31-mcp-unified-stage4l-editable-profile-crud-implementation-plan.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
