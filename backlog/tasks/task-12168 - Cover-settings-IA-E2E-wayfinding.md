---
id: TASK-12168
title: Cover settings IA E2E wayfinding
status: Done
labels:
- frontend
- settings
- e2e
- ux
documentation:
- Docs/superpowers/plans/2026-07-07-settings-ia-recovery-preferences-ui-implementation.md
modified_files:
- apps/tldw-frontend/e2e/smoke/all-pages.spec.ts
- apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add focused smoke/E2E coverage for the settings IA split: active nav count, image-gen alias active state, and compact mobile settings selector.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: `bunx playwright test e2e/smoke/stage6-interaction-stage2.spec.ts --project=chromium --reporter=line -g "settings"` passed (3 tests); `TLDW_SMOKE_HARD_GATE=0 bunx playwright test e2e/smoke/all-pages.spec.ts --project=chromium --reporter=line -g "settings route shows active location context"` passed (1 test); `git diff --check` passed. The full Stage 6 smoke file was also run and still has an unrelated pre-existing first-test failure where no `/api/v1/rag/search` call fires in `search typing and deterministic no-results answer remain functional`. `bun run typecheck` still fails in the known untouched baseline files (AudioStudio, ScheduledTasks, Skills, scheduled-tasks-control-plane, mcp-hub, voice-cloning, knowledge-qa-live, flashcards). Bandit skipped: frontend E2E TypeScript/Backlog-only changes.
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
