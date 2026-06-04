---
id: TASK-2246
title: Add governed shell facade aliases
status: Done
modified_files:
- tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py
- tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py
- tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py
- tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py
- tldw_Server_API/app/core/MCP_unified/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add shell-shaped governed command aliases after native filesystem helper tools have landed. Keep `run` as the canonical MCP tool, expose optional `bash`/`shell` facades through the same RunCommandModule implementation without raw host shell execution, and add command-runtime aliases that route `stat`, `glob`/`find`, and `rg`/`grep-files` to profile-granted MCP tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `run` remains the canonical MCP command tool.
- [ ] #2 `bash` and `shell` aliases, if exposed, call the same governed RunCommandModule implementation and clearly describe that they are not host shell execution.
- [ ] #3 Command runtime aliases route `stat <path>` to `fs.stat`, `glob <pattern> [base]` and `find <pattern> [base]` to `fs.glob`, and `rg <pattern> [base]` plus `grep-files <pattern> [base]` to `fs.grep`.
- [ ] #4 Existing pure stdin `grep` behavior remains unchanged for pipelines such as `cat app.log | grep ERROR`.
- [ ] #5 Aliases are visible/executable only when their backing MCP tools are granted to the active profile.
- [ ] #6 Tests cover policy-filtered visibility, `run --help`, alias help/error text, no raw shell delegation, and preservation of presentation footer/spill behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-04-mcp-filesystem-helper-tools-implementation-plan.md#follow-up-task-governed-shell-facade-aliases
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented governed shell facade aliases and filesystem command aliases. `run` remains canonical; `bash` and `shell` share the same RunCommandModule schema and execution path with descriptions that explicitly state they are not raw host shell surfaces. The command registry now exposes `stat`, `glob`/`find`, and `rg`/`grep-files` only when their backing filesystem MCP tools are executable, and adapters route them through prepared MCP calls. Plain `grep` remains a pure stdin transform.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added governed `bash` and `shell` compatibility tool aliases for `run`, policy-filtered filesystem command aliases for `fs.stat`, `fs.glob`, and `fs.grep`, regression coverage for visibility/routing/no-shell-delegation/pure-grep behavior, and README guidance for packaged users.

Verification:
- RED before implementation: 7 expected failures in `test_command_runtime_registry.py` and `test_run_command_module.py`.
- GREEN: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_nested_tool_preparation.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_parser.py tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_execution.py tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_presentation.py tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py -q` -> 133 passed, 4 warnings.
- GREEN: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check ...` -> all checks passed.
- GREEN: `git diff --check`.
- GREEN: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/command_runtime tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py -f json -o /tmp/bandit_mcp_governed_shell_aliases.json` -> 0 findings.
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
