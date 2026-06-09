---
id: TASK-2331
title: Harden MCP filesystem path policy and command-runtime defaults
status: Done
labels:
- mcp
- filesystem
- security
- policy
priority: High
references:
- 'User-approved approach: policy hardening first'
- runtime routing second; diff parser expansion only where current gaps block real
  use.
documentation:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-policy-hardening-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-policy-hardening-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-policy-hardening-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-policy-hardening-implementation-plan.md
- tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py
- tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py
- tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py
- tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py
- tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py
- tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the next MCP filesystem slice: verify path/action constraints for fs.read, fs.write, fs.edit, and fs.patch first, then update command-runtime defaults to prefer structured primitives over legacy fs.read_text/fs.write_text where safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-06-09-mcp-filesystem-policy-hardening-implementation-plan.md. It sequences work as: adapter candidate forwarding; action-aware path-grant regression tests; virtual CLI structured fs.read/write-create routing; focused tests, Bandit, and Backlog closeout. Plan review correction included filtered visible command descriptors so adapters can reliably choose fs.read over fs.read_text only when fs.read is actually visible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Committed implementation slices:
- 9bc89b101b fix: forward MCP path scope candidates
- 35bea990f6 test: cover MCP filesystem path grant actions
- f9bfccc5f3 feat: prefer structured MCP filesystem commands

Verification completed from worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-fs-policy-hardening`:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py -q`
  - Result: passed, 57 tests, 6 warnings in 4.50s.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py -f json -o /tmp/bandit_mcp_fs_policy_hardening.json`
  - Result: passed; JSON report at `/tmp/bandit_mcp_fs_policy_hardening.json`; findings: 0; scanned LOC: 1311.
- `git diff --check`
  - Result: passed with no output.
- `git status --short`
  - Result before Backlog closeout edit: clean.

Deferred scope:
- `write-replace` remains deferred because preimage syntax is not implemented in this slice.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
MCP filesystem policy hardening is complete for this slice. Path scope candidates are now forwarded through the policy adapter so approval grants can be evaluated against the concrete filesystem paths involved in fs.read, fs.write, fs.edit, and fs.patch operations. Regression coverage now exercises action-aware path grants, including read/write/create/patch/edit behavior and denied-scope cases.

The command runtime now prefers structured MCP filesystem commands where the server exposes them: virtual CLI read paths can route to `fs.read`, and write-create operations can route to structured write primitives instead of legacy `fs.read_text`/`fs.write_text` when safe. Visible command descriptors are filtered so structured routing is selected only when the relevant command is actually available.

`write-replace` is intentionally deferred because the required preimage syntax is not implemented in this slice.
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
