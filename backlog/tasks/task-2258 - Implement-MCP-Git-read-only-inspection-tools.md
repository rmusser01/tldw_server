---
id: TASK-2258
title: Implement MCP Git read-only inspection tools
status: In Progress
labels:
- mcp
- implementation
- git
- profiles
references:
- Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
- Docs/superpowers/plans/2026-06-04-mcp-git-read-tools-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/MCP_unified/tool_observability.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py
- tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py
- tldw_Server_API/app/core/MCP_unified/server.py
- mcp_unified/profiles/presets.py
- tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
- tldw_Server_API/app/core/MCP_unified/README.md
- mcp_unified/USER_GUIDE.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved active-workspace Git read-only MCP tools with shared tool observability metadata, optional server registration, profile grants, documentation, focused tests, adjacent regression tests, and Bandit verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Added shared MCP tool observability/evaluation metadata helpers and tests in commit cb5568787d. RED: targeted pytest failed with missing module before implementation. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py -q` -> 8 passed. `git diff --check` passed. Spec compliance review approved; code quality review approved with no findings.

Task 2 complete. Added minimal GitModule schemas/validation and tests in commit 60ae4cbec33c5a0ad1eaa444e99e59840326c357. RED: targeted pytest failed with missing `git_module.py`. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 57 passed. Spec compliance review approved; code quality review approved with no findings. Follow-up note for Task 3: ensure `execute_tool()` calls `validate_tool_arguments()` before any Git execution.
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
