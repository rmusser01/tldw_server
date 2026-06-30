---
id: TASK-170.1.1
title: Fix character row chat implicit model fallback
status: Done
assignee: []
created_date: '2026-05-09 18:43'
updated_date: '2026-05-09 18:50'
labels:
  - character-chat
  - frontend
  - ux-audit
dependencies:
  - TASK-173
documentation:
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md
  - Docs/Reviews/assets/2026-05-09-character-chat-reaudit/puppeteer-states.json
  - >-
    Docs/superpowers/plans/2026-05-09-character-row-chat-implicit-model-fallback-plan.md
parent_task_id: TASK-170.1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the remaining live character-chat row-action gap found during the post-P1 Puppeteer re-audit refresh. When no chat model has been explicitly selected, the Characters row `Chat as...` action must not treat the first fetched catalog model as a ready full-chat model. It should keep the selected character local and show the in-context model-readiness blocker instead of navigating to Companion Home. Keep quick-chat popup behavior scoped separately if it intentionally offers its own model fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A regression test covers row `Chat as...` with no `selectedModel` while the model catalog contains entries, and asserts the local model blocker appears without navigation.
- [x] #2 The row full-chat readiness path uses an explicit selected chat model rather than an implicit first catalog model fallback.
- [x] #3 Existing row-chat no-model and stale-model blocker tests continue to pass.
- [x] #4 Puppeteer/Chrome evidence confirms the returning-user row `Chat as...` action stays on Characters with the selected-character model blocker.
- [x] #5 Verification includes focused Vitest coverage, pinned UI typecheck, and doc/evidence updates from the post-P1 re-audit refresh.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a regression for the row `Chat as...` case where `selectedModel` is null but the model catalog contains entries. The test failed before the fix because the row full-chat path used the quick-chat fallback model.

Changed Characters manager wiring so `useCharacterCrud` receives explicit `selectedChatModel` for full-chat readiness; the quick-chat popup retains its separate fallback behavior.

Puppeteer/Chrome final refresh confirms `09-returning-user-row-chat-action` stays on `/characters` and shows `Character chat setup` with the selected-character model blocker.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the remaining live row-chat context leak found during the post-P1 re-audit. The Characters row `Chat as...` action no longer treats the first fetched catalog model as an implicit full-chat selection when `selectedModel` is unset; it now keeps the selected character local and shows the model-readiness blocker. Verified with a red-to-green regression, existing row no-model/stale-model tests, pinned UI typecheck, and Puppeteer/Chrome evidence.
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
