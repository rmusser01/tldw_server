---
id: TASK-12035
title: Adopt ACP Playground capability recovery state
status: Done
created_date: 2026-06-25 23:24
labels:
- webui
- acp
- ux
- accessibility
priority: medium
references:
- TASK-418.11
- TASK-214
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-06-25-webui-stage6-acp-playground-capability-recovery-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage6-acp-playground-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/ACPPlayground/index.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/ACPPlaygroundRecovery.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlaygroundRecovery.test.tsx
updated_date: 2026-06-25 23:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the standalone `/acp-playground` route. Use the existing ACP health check to render a shared user-language recovery state when ACP is unavailable, preserve the existing session/chat/tools layout when ACP is healthy or degraded, and keep raw request details behind diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP Playground shows a shared RecoveryCallout when ACP health reports unavailable before users land in a broken chat/session workspace.
- [x] #2 Recovery state includes user-language title/message, retry action, and diagnostics for method/path/status/raw message without exposing secrets.
- [x] #3 Existing ACP Playground desktop/mobile layout, deep-link handling, and session hydration remain available when ACP health is healthy or degraded.
- [x] #4 Focused tests cover healthy/degraded and unavailable ACP health paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Stage 6 plan document for the ACP Playground capability recovery slice.
2. Write a failing ACP Playground test for unavailable ACP health expecting a shared RecoveryCallout with retry and diagnostics.
3. Adjust the existing health query to retain diagnostic context and expose a retryable unavailable state without changing backend APIs.
4. Render the shared RecoveryCallout only when ACP health is definitively unavailable, preserving existing layout while loading/healthy/degraded.
5. Run focused frontend tests, lint checks, diff whitespace checks, and record browser-smoke or Bandit applicability notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a focused red/green recovery-state test for ACP unavailable health, including shared RecoveryCallout rendering, user-language copy, retry handling, and non-secret diagnostics.
- Added ACP health snapshot normalization so non-OK health responses and network failures preserve status/raw message context for diagnostics.
- Render the recovery state only when ACP health is definitively unavailable; loading, healthy, and degraded states continue through the existing ACP Playground desktop/mobile workspace.
- Cleaned hook dependency lint warnings in the touched ACP Playground file by stabilizing the fallback element and pending-permissions fallback.
- Verification: `bun run test:run ../packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlaygroundRecovery.test.tsx` passed with 2 tests; touched-file ESLint exited 0 with only the existing repo-level Next pages-directory warning; `git diff --check` passed.
- Known local harness limitation: the broader ACP Playground connection suite was not extended because Vite resolves `ACPChatPanel` through the UI package path and fails before test execution on missing `rehype-highlight`; recovery behavior is covered with the isolated component test and the index wiring was linted.
- Bandit: not applicable for this TS/TSX/docs-only slice.
- Final self-review tightened ACP health normalization so HTTP/network failure fallbacks override any misleading `overall` value in a response payload; the focused recovery test now covers that edge case.
- Re-verification after self-review: focused Vitest passed with 3 tests; touched-file ESLint exited 0 with only the existing repo-level Next pages-directory warning; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the ACP Playground capability recovery slice: unavailable ACP health now maps to a shared retryable RecoveryCallout with method/path/status/raw-message diagnostics, while healthy/degraded/loading health preserves the existing workspace. Added focused regression coverage for hidden healthy/degraded paths, forced-unavailable normalization for failed health responses, and unavailable recovery rendering; recorded the local broader-suite dependency-resolution limitation.
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
