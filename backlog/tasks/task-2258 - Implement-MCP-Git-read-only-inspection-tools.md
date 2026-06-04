---
id: TASK-2258
title: Implement MCP Git read-only inspection tools
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 19:20'
labels:
  - mcp
  - implementation
  - git
  - profiles
dependencies: []
references:
  - Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
  - Docs/superpowers/plans/2026-06-04-mcp-git-read-tools-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved active-workspace Git read-only MCP tools with shared tool observability metadata, optional server registration, profile grants, documentation, focused tests, adjacent regression tests, and Bandit verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Added shared MCP tool observability/evaluation metadata helpers and tests in commit cb5568787d. RED: targeted pytest failed with missing module before implementation. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py -q` -> 8 passed. `git diff --check` passed. Spec compliance review approved; code quality review approved with no findings.

Task 2 complete. Added minimal GitModule schemas/validation and tests in commit 60ae4cbec33c5a0ad1eaa444e99e59840326c357. RED: targeted pytest failed with missing `git_module.py`. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 57 passed. Spec compliance review approved; code quality review approved with no findings. Follow-up note for Task 3: ensure `execute_tool()` calls `validate_tool_arguments()` before any Git execution.

Task 3 complete. Added bounded async Git runner, safe Git env, repo discovery, structured repository-prep errors, and tests in commits 767c923158377908ff4c7526d71910a8715f32b4 and 820f12a08b080f29a4706f005286fa1083774b38. RED: runner/repository selector failed before implementation/fix. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"` -> 20 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 57 passed; full `test_git_module.py` -> 77 passed. `git diff --check` passed. Bandit on touched implementation reported 0 findings. Initial code review found unbounded subprocess output and permissive global options; fixes added bounded pipe reads, truncation, process kill on output cap overflow, and unsafe global option rejection. Spec compliance and code quality re-reviews approved.
Task 4 complete. Implemented `git.status`, `git.branches`, and `git.conflicts.list` in commits cafa9cac38c0a88408bfb6654867eef239bcd059, 1e59309c017032d2ae791f49d6b1a210bd85c6a9, and f9aa618b42a6f52ebc9d01a8c48b0bf2cdbc8e7b. RED: status/branches/conflicts selector initially failed because the tools still returned `not_implemented`; follow-up RED checks caught status response nesting and delete/modify conflict status mapping. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "status or branches or conflicts_list"` -> 18 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"` -> 20 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 57 passed; full `test_git_module.py` -> 84 passed. `git diff --check` passed. Bandit on touched implementation reported 0 findings. Spec compliance re-review approved. Code quality re-review approved after correcting `{1,2}`/`{1,3}` delete-modify conflict mapping.
Task 5 complete. Implemented `git.diff`, `git.log`, `git.blame`, and `git.conflicts.read` in commits fe31eac395b145faaa10b5faabb6f1cb17008af7, 8a71fed35862e0ac18fdfc0910139c68cca496e1, 1d8edb365c1a0841e62e7e4be60938f0de8ca7cd, 07fdde15acea61561459d729e05c2929811db939, 815a2ad3b77a2e8283885486298b4c23cf6a7a11, and a5a1f811359218e832af588723093de5b43a88a1. RED: diff/log/blame/conflicts-read selector initially failed because tools returned `not_implemented`; follow-up RED checks covered structured path refusals, payload shape fields, nested repo path translation, stable reason-code mapping, working-tree diff byte bounds, and root-backslash path validation. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "diff or log or blame or conflicts_read"` -> 60 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "status or branches or conflicts_list"` -> 18 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or repository or timeout"` -> 20 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "schema or validates"` -> 61 passed; full `test_git_module.py` -> 111 passed. `git diff --check` passed. Bandit on touched implementation reported 0 findings. Spec compliance re-review approved; code quality review approved.
Task 6 complete. Registered `GitModule` behind `MCP_ENABLE_GIT_MODULE` and granted the full native Git read tool set to Git-capable presets in commits dd21e7150ee50a209de44bac2de0b1b5e6d58111 and 4a3017db3f0ad5f883fa581bf0301be75bdb4145. RED: registration test failed because `MCP_ENABLE_GIT_MODULE=1` did not register Git; profile preset tests failed because Git-capable presets lacked the full native tool set; follow-up RED checks covered duplicate git registration and progressive-disclosure direct-tool limits. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py -q` -> 5 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q` -> 29 passed; `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q` -> 111 passed. `git diff --check` passed. Bandit on production touched files plus the new registration test passed; broader profile test file still has existing low-severity B101 assert baseline findings. Spec compliance re-review approved; code quality re-review approved after bounding direct Git exposure for overloaded presets and adding the duplicate guard.
Task 7 complete. Documented optional Git read-only inspection tools in commit 3bc95a86a1d77f7b8625681909f9d441d7c26ddd. Updated the MCP README with `MCP_ENABLE_GIT_MODULE=true`, the seven Git tools, read-only/no-shell guarantees, active-workspace repository binding, ignored-file/email/diff-helper safety notes, bounded responses, eval metadata, and follow-up `TASK-2256`. Updated the packaged user guide with the Git inspection tools subsection near profile/tool discovery guidance, including enabled profiles and Product Owner/Documentation Writer exclusions. `git diff --check` passed. Spec compliance review approved; documentation quality review approved.
Final review fix complete. Addressed whole-branch review findings in commit 9b97b0bdb9eba29d17e09504ee8c92896e763f5b by neutralizing repo-local `core.fsmonitor` through runner `GIT_CONFIG_*` environment overrides and adding `--no-textconv` to `git.blame`. RED: targeted tests failed for missing fsmonitor override and blame textconv guard. GREEN: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py -q -k "runner or status or blame"` -> 46 passed; full `test_git_module.py` -> 111 passed. Focused review approved and live probes on Apple Git 2.39.5 confirmed fsmonitor/textconv helpers are neutralized.

Final verification at head 9b97b0bdb9eba29d17e09504ee8c92896e763f5b: focused suite `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module.py tldw_Server_API/app/core/MCP_unified/tests/test_git_module_registration.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q` -> 148 passed, 4 warnings. Adjacent MCP regressions `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_browser_cdp_server_registration.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q` -> 63 passed, 5 warnings. Bandit source scan `python -m bandit -r tldw_Server_API/app/core/MCP_unified/tool_observability.py tldw_Server_API/app/core/MCP_unified/modules/implementations/git_module.py tldw_Server_API/app/core/MCP_unified/server.py mcp_unified/profiles/presets.py -f json -o /tmp/bandit_mcp_git_read_tools.json` -> 0 findings. `git diff --check` passed. `git status --short --branch` clean, branch ahead of origin/dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

PR #2266 review pass complete after rebasing onto latest origin/dev. Verified and fixed three still-valid review threads: Git runner no longer silently suppresses broad cleanup exceptions and logs unexpected process-wait failures; Git read commands now use separate git_command_timeout_seconds instead of repository discovery timeout; blame parsing caches author metadata by commit hash for repeated --line-porcelain commits. Also fixed the failing onboarding docs gate by restoring required user-guide index discoverability entries for benchmark, OpenWebUI import/hydration, and flashcards in source and published indexes. Validation: new RED tests failed for all three review findings before implementation; after fixes test_git_module.py -> 114 passed, related MCP/profile tests -> 37 passed, docs discoverability tests -> 9 passed, Bandit touched source -> 0 findings, git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the active-workspace read-only MCP Git inspection tool family and supporting metadata, registration, profile grants, tests, and docs. The branch adds shared tool observability metadata, a bounded async Git runner with safe environment and fixed argv controls, seven read-only Git tools, optional `MCP_ENABLE_GIT_MODULE` registration, Git-capable profile grants, README/user-guide documentation, and Backlog/spec/plan records. Final verification passed focused and adjacent MCP test suites, Bandit reported zero findings on touched source, and final review hardening neutralized repo-local fsmonitor/textconv helper execution.
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
