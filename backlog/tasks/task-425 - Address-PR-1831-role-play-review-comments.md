---
id: TASK-425
title: Address PR 1831 role-play review comments
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/1831
- https://github.com/rmusser01/tldw_server/pull/1831#discussion_r3255579703
- https://github.com/rmusser01/tldw_server/pull/1831#discussion_r3255579705
- https://github.com/rmusser01/tldw_server/pull/1831#discussion_r3255579707
- https://github.com/rmusser01/tldw_server/pull/1831#discussion_r3255597555
- https://github.com/rmusser01/tldw_server/pull/1831#discussion_r3255597560
documentation:
- apps/tldw-frontend/output/playwright/pr-1831-character-chat-ui.png
modified_files:
- apps/packages/ui/src/components/Common/PromptSelect.tsx
- apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePromptTemplates.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/usePromptTemplates.role-play-apply.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve current actionable review feedback on PR #1831 for the main /chat role-play preset remediation branch. Scope covers saved role-play setup template identity preservation, persona setup restore, prompt recovery menu availability, verification of Gemini false positives, focused tests, and CDP screenshot capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid PR #1831 review comments are verified against the rebased branch and either fixed or documented as already addressed.
- [x] #2 Role-play setup apply/clear paths do not leave stale persona, character, behavior template, scene, or generation state.
- [x] #3 New or changed role-play controls use localized user-facing labels and avoid brittle source-string-only tests where behavior tests are feasible.
- [x] #4 Focus, async error handling, clipboard failure handling, and malformed scene draft recovery paths are covered by implementation or focused tests.
- [x] #5 Focused role-play and review-fix tests are run, with any unrelated baseline failures documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1831 review feedback by preserving saved role-play behavior template IDs on apply, restoring saved persona setups through selectedAssistant, keeping current system-prompt recovery actions visible when saved prompts exist, and verifying Gemini's documentContext/updateChatModelSettings comments as false positives. Verification: focused Vitest review-fix tests passed (14 tests), broader focused role-play suite passed (15 files, 134 tests), git diff --check passed, CDP screenshot captured at apps/tldw-frontend/output/playwright/pr-1831-character-chat-ui.png. TypeScript check still fails only on existing dev-baseline files outside this slice; Bandit skipped because no Python files were touched.
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
