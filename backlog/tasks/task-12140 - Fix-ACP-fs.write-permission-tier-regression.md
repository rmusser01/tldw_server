---
id: TASK-12140
title: Fix ACP fs.write permission tier regression
status: Done
assignee: []
created_date: '2026-07-04 02:10'
updated_date: '2026-07-04 03:15'
labels:
  - tests
  - acp
  - permissions
dependencies: []
references:
  - >-
    /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory/backlog/tasks/task-12139
    - Fix-pytest-app-startup-MCP-validation-runtime-blocker.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the next broad pytest blocker after MCP startup validation was repaired: ACP websocket permission-tier test expects fs.write to resolve to the batch tier, but current code returns individual.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused ACP permission-tier test passes.
- [x] #2 Root cause is documented and the fix preserves intended individual/batch operation boundaries.
- [x] #3 Broader stop-on-first-failure pytest retry outcome is recorded after the fix.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: test_acp_websocket.py still expected fs.write and modify_file to be batch operations, but shared ACP permission tier logic intentionally classifies write/modify tokens as individual after the runner hardening work. The fix updates websocket expectations to keep fs.write and modify_file individual, while batch coverage now uses policy-neutral examples: artifact.save, git.commit, and workspace.plan. Also annotated the test-only token literal with nosec B106 so touched-scope Bandit remains clean.

Verification: focused pytest for stream-start cleanup, websocket permission-tier individual/batch checks, and sandbox standard-runner tier comparison passed: 4 passed, 16 warnings. Bandit on test_acp_websocket.py with B101 skipped exited 0. git diff --check exited 0. Broad pytest -q -x --tb=short passed the prior ACP websocket blocker, completed test_acp_websocket.py, continued through Audio and Audit into AuthNZ integration tests, then was manually stopped to avoid a long full-suite run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned ACP websocket permission-tier tests with current runner semantics: write/modify operations are individual approvals, while non-destructive policy-neutral operations remain batch. Focused tests, Bandit, and diff check passed; the broad retry cleared the prior ACP blocker and continued into unrelated suites before being stopped.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused regression verification recorded.
- [x] #2 Bandit run for touched Python code recorded.
- [x] #3 Modified files recorded.
- [x] #4 Final summary added.
<!-- DOD:END -->
