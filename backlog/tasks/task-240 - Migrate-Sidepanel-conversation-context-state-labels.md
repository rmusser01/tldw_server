---
id: TASK-240
title: Migrate Sidepanel conversation context state labels
status: Done
assignee: []
created_date: '2026-05-10 19:20'
updated_date: '2026-05-10 19:33'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1544'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the frontend design-system product-state cleanup by replacing Sidepanel conversation-context readiness hardcoded Ready and Blocked labels with canonical design-system state registry labels while preserving existing readiness/tone behavior. Current dev also has bounded MCPHub product-state baseline drift, so this task refreshes those baseline records to keep the verifier green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 resolveContextReadiness uses the design-system ready and blocked state registry labels for ready and blocked readiness states.
- [x] #2 Focused utility coverage proves ready and blocked labels are supplied through the registry rather than hardcoded literals.
- [x] #3 The matching conversation-context-utils Ready and Blocked baseline exceptions are removed and the design-system verifier passes.
- [x] #4 Current dev MCPHub product-state baseline drift is reconciled without reintroducing the conversation-context Ready or Blocked exceptions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red: conversation-context-utils test mocked ready and blocked registry labels as 'Ready via registry'/'Blocked via registry' and failed while resolveContextReadiness still returned literal Ready.

Green verification after rebase onto origin/dev: conversation-context-utils focused Vitest passed (1 test); product-state guard unit test passed (52 tests); verify:design-system-state exited 0; git diff --check HEAD~1..HEAD exited 0. Broad bunx tsc still exits 2 with 239 baseline errors, and touched-scope filtering found no conversation-context/baseline/task/MCPHub matches.

Bandit skipped: touched implementation is TypeScript/TSX/JSON/Backlog only, with no Python runtime code.

Addressed PR review feedback for runtime registry label reads. Added regression coverage for mocked design-system label updates after import and moved ready/blocked label lookups into resolveContextReadiness so labels are resolved at call time. Focused Vitest now passes with 2 tests.

Review-fix verification: focused conversation-context-utils Vitest passed with 2 tests, product-state guard passed with 52 tests, verify:design-system-state exited 0, git diff --check exited 0. Broad bunx tsc still exits 2 with 239 known baseline errors; touched-scope filter found no matches for conversation-context-utils, design-system baseline, task-240, MCPHub baseline files, or getDesignSystemState.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
resolveContextReadiness now uses getDesignSystemState('ready').label and getDesignSystemState('blocked').label for canonical ready and blocked readiness states, with a mocked-registry unit test proving the labels are not hardcoded. Removed the conversation-context-utils Ready/Blocked baseline exceptions and refreshed bounded current-dev MCPHub baseline drift so the design-system verifier exits 0.

PR: https://github.com/rmusser01/tldw_server/pull/1544

PR review follow-up: resolveContextReadiness now reads canonical ready/blocked labels at resolution time rather than caching them at module load, with regression coverage for runtime registry label updates.
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
