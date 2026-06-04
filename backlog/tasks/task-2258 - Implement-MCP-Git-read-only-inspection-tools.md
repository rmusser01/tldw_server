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

Task 3 complete. Added bounded async Git runner, safe Git env, repo discovery, structured repository-prep errors, and tests in commits 767c923158377908ff4c7526d71910a8715f32b4 and 820f12a08b080f29a4706f005286fa1083774b38. RED: runner/repository selector failed before implementation/fix. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"` -> 20 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 57 passed; full `test_git_module.py` -> 77 passed. `git diff --check` passed. Bandit on touched implementation reported 0 findings. Initial code review found unbounded subprocess output and permissive global options; fixes added bounded pipe reads, truncation, process kill on output cap overflow, and unsafe global option rejection. Spec compliance and code quality re-reviews approved.
Task 4 complete. Implemented `git.status`, `git.branches`, and `git.conflicts.list` in commits cafa9cac38c0a88408bfb6654867eef239bcd059, 1e59309c017032d2ae791f49d6b1a210bd85c6a9, and f9aa618b42a6f52ebc9d01a8c48b0bf2cdbc8e7b. RED: status/branches/conflicts selector initially failed because the tools still returned `not_implemented`; follow-up RED checks caught status response nesting and delete/modify conflict status mapping. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "status or branches or conflicts_list"` -> 18 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"` -> 20 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 57 passed; full `test_git_module.py` -> 84 passed. `git diff --check` passed. Bandit on touched implementation reported 0 findings. Spec compliance re-review approved. Code quality re-review approved after correcting `{1,2}`/`{1,3}` delete-modify conflict mapping.
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
