---
id: TASK-2379
title: Implement MCP configurable tool-call hook manager integration
status: Done
labels:
- mcp
- gateway
- hooks
priority: medium
modified_files:
- mcp_unified/tool_hooks/__init__.py
- mcp_unified/tool_hooks/manager.py
- mcp_unified/tool_hooks/models.py
- mcp_unified/tool_use_reporting/models.py
- mcp_unified/tool_use_reporting/__init__.py
- mcp_unified/pyproject.toml
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_hook_manager.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
- mcp_unified/README.md
- mcp_unified/USER_GUIDE.md
- Docs/superpowers/plans/2026-06-17-mcp-configurable-tool-call-hooks-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first implementation slice after the MCP tool-call hook seam: a configurable/runtime hook manager that can register ordered pre/post hooks, preserve fail-closed governance semantics, expose decisions to audit/reporting, and remain host-neutral for standalone gateway use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Runtime can be constructed with a concrete hook manager/registry without requiring host-specific code.
- [ ] #2 Configured hooks execute in deterministic order for pre and post tool-call phases.
- [ ] #3 Pre-hook failures fail closed with governance errors and do not execute the tool.
- [ ] #4 Post-hook failures are observed/audited without changing the original tool result or error.
- [ ] #5 Hook decisions and failures surface in existing tool-call audit/reporting metadata without logging sensitive payloads.
- [ ] #6 Focused tests cover default no-op behavior, ordering, deny/failure handling, and post-hook non-interference.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-17-mcp-configurable-tool-call-hooks-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a package-level configurable MCP tool-call hook manager with ordered pre/post registrations, first-blocking pre-hook behavior, pre-hook fail-closed errors, and post-hook failure continuation. Added metadata-only hook result rows to tool-use reporting and wired protocol hook decisions/failures into tool-use events without raw hook messages, callback metadata, tool arguments, or absolute paths. Updated package docs and manifest entries for the new hook package and reporting package. Validation after rebasing onto latest origin/dev: 212 focused MCP package/reporting/protocol tests passed; production Bandit passed with zero findings; full touched-scope Bandit only reported pytest B101 assert warnings; wheel build succeeded with a pre-existing setuptools license deprecation warning.
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
