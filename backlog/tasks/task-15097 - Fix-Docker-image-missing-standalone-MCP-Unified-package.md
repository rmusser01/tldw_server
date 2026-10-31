---
id: TASK-15097
title: Fix Docker image missing standalone MCP Unified package
status: Done
modified_files:
- Dockerfiles/Dockerfile.prod
- tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users following the single-user Docker guide can hit ModuleNotFoundError for the standalone mcp_unified package because Dockerfile.prod installs the root project before copying apps/mcp-unified/src into the builder context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-12175 to TASK-15097 on 2026-10-31 (TASK-13960): TASK-12175 now names only "Phase 2.2 collections reading router conditional cleanup L" (backlog/tasks/task-12175 - Phase-2.2-collections-reading-router-conditional-cleanup-L.md). Before 2026-10-31 both tasks used TASK-12175; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the single-user Docker image path so the standalone MCP Unified source is present when pip installs the root package. Verified with the targeted MCP Docker packaging contract test and Bandit on the touched test file.
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
