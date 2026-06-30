---
id: TASK-159
title: Implement character-chat model readiness blockers
status: Done
assignee: []
created_date: '2026-05-09 05:34'
updated_date: '2026-05-09 05:54'
labels:
  - character-chat
  - frontend
  - ux-audit
  - model-readiness
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-09-character-chat-model-readiness-plan.md
  - Docs/superpowers/specs/2026-05-09-character-chat-ux-work-packages-design.md
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the character-chat model readiness work package: define a shared readiness contract for starting character chat, replace fragmented no-model/provider blocker language in character-chat surfaces, and verify no-model/model-ready behavior without losing selected-character context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A shared typed readiness helper distinguishes server availability, selected character, configured model/provider, and chat-send readiness.
- [x] #2 Existing model-loading or validation utilities are reused rather than duplicating provider/model logic.
- [x] #3 Character-chat no-model/provider blockers use consistent in-context language and do not redirect users away from their selected-character task by default.
- [x] #4 Character generation clearly treats AI generation model availability as separate from required character-chat send readiness.
- [x] #5 Focused unit/component tests cover missing-model, missing-character, and ready states.
- [x] #6 Verification commands are recorded, including frontend tests and any relevant lint/type checks that can run in the local environment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented shared character-chat readiness helper in apps/packages/ui/src/utils/chat-model-availability.ts and reused existing model normalization/catalog helpers for model availability checks.

Added in-context quick-chat no-model blocker in CharacterDialogs/useCharacterQuickChat, preserved selected character context, and linked directly to /settings/model instead of redirecting away from the selected-character task.

Updated character generation no-model copy to distinguish optional AI generation from saved-character chat availability; added focused GenerateCharacterPanel coverage.

Verification: ./node_modules/.bin/vitest run src/utils/__tests__/chat-model-availability.test.ts --maxWorkers=1 --no-file-parallelism passed with 12 tests.

Verification: ./node_modules/.bin/vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx --testNamePattern "shows an in-context blocker when quick chat has no chat model" --maxWorkers=1 --no-file-parallelism passed with 1 focused test and 86 skipped.

Verification: ./node_modules/.bin/vitest run src/components/Option/Characters/__tests__/GenerateCharacterPanel.test.tsx --maxWorkers=1 --no-file-parallelism passed with 1 test.

Verification: git diff --check passed.

Typecheck: ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json currently fails on existing broad UI type debt; new helper-specific TFunction diagnostics were fixed. Remaining filtered diagnostics are pre-existing in large files such as PlaygroundChat, CharacterDialogs parse-error rendering, and older Manager mock tuple typing.

Bandit: not applicable because this package only changes frontend TypeScript/React code and docs/backlog records.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the character-chat model readiness package. Added a shared readiness contract and copy helper, wired it into character quick chat and selected-character chat no-model messaging, kept AI character generation model availability separate from chat send readiness, and added focused utility/component tests. Package-wide TypeScript remains blocked by existing UI baseline errors, but the new helper-specific diagnostics were resolved and focused Vitest checks plus git diff hygiene passed.
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
