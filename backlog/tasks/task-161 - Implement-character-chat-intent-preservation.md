---
id: TASK-161
title: Implement character-chat intent preservation
status: Done
assignee: []
created_date: '2026-05-09 06:02'
updated_date: '2026-05-09 06:13'
labels:
  - character-chat
  - frontend
  - ux-audit
  - intent-preservation
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-09-character-chat-intent-preservation-plan.md
  - Docs/superpowers/specs/2026-05-09-character-chat-ux-work-packages-design.md
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the character-chat intent-preservation work package: preserve the selected character across row-level Chat actions, model-readiness blockers, and setup navigation so users can return to the intended character chat instead of losing context to a generic fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Row-level Chat as character records the intended character before any setup or navigation step.
- [x] #2 Missing model state shows a local blocker naming the selected character with actions for model setup, retry, and return to the character.
- [x] #3 Selected-character intent is preserved across model setup navigation and can be cleared explicitly to avoid stale leakage.
- [x] #4 With a configured model, the row action opens or creates the intended character-chat flow without falling back to generic Companion Home.
- [x] #5 Focused tests cover no-model preservation, model-ready behavior, and switching/clearing between characters.
- [x] #6 Verification commands are recorded, including frontend tests and any available lint/type checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented selected-character intent preservation for row-level full character chat. The Characters row action now stores the selected character before readiness checks and uses the shared character-chat readiness contract before navigating.

Added a local character-chat setup blocker in CharacterDialogs for missing chat-model readiness. It names the selected character and offers Open model settings, Retry character chat, and Return to character actions.

Explicit Return to character / close clears the selected-character handoff to avoid stale intent leakage. The model settings link preserves the selected character so setup can return to the intended chat context.

Preserved model-ready behavior: with a mocked model catalog, first-run template create -> chat handoff still stores the selected character, navigates to /, and focuses the composer. Empty server chat creation remains deferred until first send.

Verification: ./node_modules/.bin/vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx --testNamePattern "supports first-run template -> create -> chat handoff|shows an in-context blocker when quick chat has no chat model|preserves row chat intent locally when no chat model is configured|promotes quick chat into full chat flow without route changes until promoted" --maxWorkers=1 --no-file-parallelism passed with 4 tests.

Verification: ./node_modules/.bin/vitest run src/utils/__tests__/chat-model-availability.test.ts --maxWorkers=1 --no-file-parallelism passed with 12 tests.

Verification: git diff --check passed.

Typecheck: ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json remains blocked by known UI baseline errors. Filtered output for touched files still reports pre-existing tuple mock typing, CharacterDialogs parse-error translation typing, notification duration typing, and tag option typing; no new diagnostics point at the intent-blocker lines.

Bandit: not applicable because this package only changes frontend TypeScript/React code and docs/backlog records.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the character-chat intent preservation package. Row-level Chat as character now preserves selected-character intent, blocks locally when chat-model readiness fails, offers setup/retry/return actions, clears stale intent on explicit return, and preserves the existing model-ready handoff. Focused Vitest checks and diff hygiene passed; package-wide TypeScript remains blocked by existing UI baseline errors.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Bandit run for touched backend code when applicable or document frontend-only skip
- [x] #8 Implementation plan updated with outcomes and known blockers
<!-- DOD:END -->
