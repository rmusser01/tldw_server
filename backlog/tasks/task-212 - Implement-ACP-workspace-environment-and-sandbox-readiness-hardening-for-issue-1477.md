---
id: TASK-212
title: >-
  Implement ACP workspace environment and sandbox readiness hardening for issue
  1477
status: Done
assignee: []
created_date: '2026-05-10 02:14'
updated_date: '2026-05-10 02:26'
labels:
  - ACP
  - sandbox
  - workspace
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1477'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Development/ACP_Production_Readiness.md
  - Docs/Development/Agent_Client_Protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the #1477 sandbox/workspace readiness slice in the ACP productionization worktree. Existing tests already cover workspace allowlist basics and sandbox hardening defaults, but dispatch claims workspace env vars are merged into ACP sessions while only MCP servers are currently passed. Add focused red/green tests for workspace env propagation, clearer workspace validation failure payloads, sandbox runner env merge behavior, and documentation for runtime diagnostics/caveats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace path validation failures return stable actionable error payloads.
- [x] #2 Workspace MCP server injection and workspace env propagation are covered by backend tests.
- [x] #3 Standard and sandbox ACP session creation accept per-session env without dropping existing config env.
- [x] #4 Sandbox/workspace production caveats and operator guidance are documented for #1477.
- [x] #5 GitHub issue #1477 is updated with implementation status and verification evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Scope and Contract
**Goal**: Define the narrow #1477 backend hardening contract.
**Success Criteria**: Tests and docs target workspace validation errors, MCP/env session injection, and sandbox env merge behavior.
**Tests**: Focused workspace helper and sandbox runner tests.
**Status**: Complete

## Stage 2: Red Tests
**Goal**: Add failing tests for actionable workspace errors, dispatch env propagation, and sandbox session env merge.
**Success Criteria**: Tests fail on missing structured error detail or missing session env propagation.
**Tests**: Targeted Agent Orchestration and ACP sandbox tests.
**Status**: Complete

## Stage 3: Backend Hardening
**Goal**: Implement structured validation details and per-session env propagation through standard and sandbox ACP session creation.
**Success Criteria**: MCP server injection remains unchanged and env propagation is additive.
**Tests**: Targeted tests pass.
**Status**: Complete

## Stage 4: Docs and Verification
**Goal**: Update ACP docs/readiness row, run focused tests, broader relevant suites, Bandit, diff check, then update #1477.
**Success Criteria**: Backlog and GitHub issue include evidence.
**Tests**: Focused pytest, relevant suites, Bandit, git diff check.
**Status**: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented stable workspace validation error payloads, workspace env propagation through orchestration dispatch, standard runner session env forwarding, and sandbox config/session env merge via ACP_AGENT_ENV_JSON.

Verification so far:
- Red targeted #1477 tests failed as expected: 5 failed.
- Targeted #1477 tests: 5 passed.
- Focused workspace/orchestration/sandbox files: 65 passed.
- Agent Orchestration suite: 151 passed.
- ACP runtime-policy/sandbox/Lima focused suite: 28 passed.
- Full Agent_Client_Protocol directory: 809 passed, 3 failed in test_acp_schedules.py; failures are isolated to #1474 schedules/triggers and did not touch the #1477 paths.
- Bandit touched backend Python: results=[], loc=6435, output /tmp/bandit_acp_workspace_sandbox_1477.json.
- git diff --check: clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the #1477 backend hardening slice: structured workspace root validation errors, workspace env propagation through orchestration dispatch, standard runner per-session env forwarding, sandbox config/session env merging through ACP_AGENT_ENV_JSON, and operator documentation/readiness updates.

Verification: targeted red tests failed first; targeted #1477 tests passed; focused workspace/orchestration/sandbox files passed (65); Agent Orchestration passed (151); ACP runtime-policy/sandbox/Lima focused suite passed (28); Bandit found no issues across touched backend Python; git diff --check was clean. Full Agent_Client_Protocol has 3 existing/out-of-scope schedule failures in test_acp_schedules.py for #1474.
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
