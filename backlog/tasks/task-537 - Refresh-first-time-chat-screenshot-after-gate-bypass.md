---
id: TASK-537
title: Refresh first-time chat screenshot after gate bypass
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 03:22'
labels:
  - chat
  - ux
  - evidence
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh the first-time /chat screenshot and evidence after TASK-536 changed /chat to bypass the global assistant setup modal. Keep scope limited to live evidence capture and evidence metadata/docs; do not add new /chat behavior unless the screenshot path exposes a blocker that must be fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh first-time /chat screenshot shows the post-TASK-536 route behavior without the global assistant setup overlay.
- [x] #2 Evidence JSON and review notes reference the refreshed screenshot as current state.
- [x] #3 Focused verification proves the app-shell route contract still passes.
- [x] #4 Any backend/dev-server startup limitation is diagnosed or recorded with concrete evidence.
- [x] #5 Bandit applicability and final verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-28-chat-first-run-screenshot-refresh.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-537 refreshed first-time /chat evidence after TASK-536. Backend startup was diagnosed as an environment issue: sandboxed bind to 127.0.0.1:18041 failed with [Errno 1] operation not permitted, while the same command succeeded with approved elevated localhost binding and /api/v1/health returned ok in single_user mode.

Captured first-time-unseeded.png from a clean Playwright Chromium context at http://localhost:18042/chat. Assertions: route stayed /chat, first-run-gate-overlay count 0, Build Your Assistant copy count 0, chat input count 1, Start a new chat heading count 1. Screenshot is 1440x960 PNG and shows context/runtime rails on the current first-time chat surface.

Verification: bunx vitest run __tests__/app/app-layout.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx passed: 2 files, 17 tests. evidence.json parsed. git diff --check passed. Ports 18041 and 18042 were stopped and no longer listen. Bandit skipped because TASK-537 touched Markdown, JSON, and PNG evidence only; no Python code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the current first-time /chat screenshot and evidence after the first-run gate bypass. Updated the evidence JSON, review doc, asset README, and plan so first-time evidence now reflects the live chat cockpit state instead of the old global assistant setup overlay. Recorded the sandbox bind limitation, the elevated successful backend path, focused route-contract verification, JSON/whitespace checks, port cleanup, and Bandit skip rationale.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python code or documented skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
