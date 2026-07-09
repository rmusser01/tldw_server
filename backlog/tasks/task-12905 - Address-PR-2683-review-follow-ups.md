---
id: TASK-12905
title: Address PR 2683 review follow-ups
status: Done
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2683
modified_files:
- apps/packages/ui/src/services/acp/connection.ts
- apps/packages/ui/src/services/acp/__tests__/connection.test.ts
- apps/packages/ui/src/components/Common/Playground/Message.tsx
- apps/packages/ui/src/components/Common/Playground/__tests__/Message.routing-fallback.integration.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-valid top-level PR review findings after rebasing the settings IA split PR onto dev. Keep fixes minimal and scoped to verified issues only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ACP single-user auth ignores scrubbed, whitespace, or placeholder stored keys and falls back to the runtime override.
- [ ] #2 Mood badge display normalizes hyphenated and underscored mood labels consistently.
- [ ] #3 Regression tests cover the changed behavior.
- [ ] #4 Touched-scope verification passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify review findings against the current code and PR diff.
2. Apply minimal code fixes for still-valid findings.
3. Add or update focused tests.
4. Run targeted frontend tests and diff checks.
5. Update the PR/backlog with the outcome.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the still-valid top-level PR review findings with minimal frontend changes and regression tests. Verification passed: focused ACP/Message Vitest suites, settings-specific Vitest suite, settings alias/mobile Playwright slice, settings stage 6 Playwright slice, and git diff --check. Bandit is not applicable because the touched implementation files are TypeScript/TSX, not Python.
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
