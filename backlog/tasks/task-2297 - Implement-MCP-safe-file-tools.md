---
id: TASK-2297
title: Implement MCP safe file tools
status: To Do
labels:
- mcp
- filesystem
- security
- implementation
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
modified_files:
- Docs/superpowers/plans/2026-06-07-mcp-safe-file-tools-implementation-plan.md
- mcp_unified/interfaces/path_scope.py
- mcp_unified/interfaces/policy.py
- tldw_Server_API/app/core/MCP_unified/modules/base.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_diff.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_receipts.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- mcp_unified/profiles/presets.py
- Docs/MCP_UNIFIED_USER_GUIDE.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved MCP safe file tools design: action-aware path grants, module-derived path candidates, fs.read with hashes/read receipts, fs.patch unified-diff editing, fs.write guarded create/replace behavior, legacy tool compatibility metadata, profile guidance updates, observability redaction, tests, and docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-07-mcp-safe-file-tools-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
