---
id: TASK-12064
title: Rebase PR 2548 and verify MCP residual UX review follow-ups
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-29 20:17'
labels:
  - pr-2548
  - mcp
  - ci
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track rebasing PR #2548 onto latest dev, verifying all PR review comments/issues are addressed, investigating failing PR checks, and recording final verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto latest origin/dev
- [x] #2 All GitHub review threads and top-level comments have been checked and actionable issues addressed or documented
- [x] #3 Relevant local tests and Bandit on touched scope have been run
- [x] #4 PR branch is pushed and PR status is reported
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Fetched latest `origin/dev` and PR head, then rebased `codex/mcp-residual-ux-clean` onto `origin/dev`.
- Verified the existing PR review fixes after rebase: package gateway readiness uses `package_summary.get(...)`, admin-auth `None` is handled, `/mcp/status` has `GatewayReadinessStatusResponse`, and MCP status distinguishes configured-but-not-loaded modules.
- Queried review threads and top-level comments; no unresolved review threads remained at the time of inspection.
- Investigated old PR check failures. The README release metadata failure is already fixed on rebased `dev`; local contract test passes.
- Root-caused the MCP authz CI failures to tests assuming the high-risk `run_command` module is loaded by default after this PR intentionally made it explicit opt-in.
- Updated `tldw_Server_API/tests/MCP/test_mcp_tools_execute_authz.py` so run-command authz tests write a temporary explicit MCP modules YAML and set `MCP_MODULES_CONFIG` via `monkeypatch` before resetting the MCP server.

Verification:
- Targeted red check before the fix reproduced the two `Tool not found: run` failures.
- Targeted rerun after the fix: `2 passed, 3 warnings`.
- Full authz file: `4 passed, 3 warnings`.
- MCP focused suite: `270 passed, 7 warnings`.
- Release docs metadata contract: `1 passed, 3 warnings`.
- CI-style MCP shard: `577 passed, 4 skipped, 975 warnings`.
- `git diff --check`: passed.
- Bandit on touched MCP/app/test scope wrote `/tmp/bandit_pr2548_mcp_scope.json`; only LOW-severity B101 assert findings in the touched pytest file, no production-code findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2548's `codex/mcp-residual-ux-clean` branch onto latest `origin/dev`, verified the GitHub review feedback remains addressed, and fixed the PR-specific MCP authz CI failure by making the run-command tests explicitly opt into the high-risk `run_command` module. Local verification passed for the focused authz tests, MCP focused suite, release docs contract, CI-style MCP shard, whitespace check, and Bandit touched-scope scan with only test assert warnings.
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
