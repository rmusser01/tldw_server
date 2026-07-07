---
id: TASK-12906
title: Implement character expression editor onboarding
status: In Progress
labels:
- frontend
- characters
- emotes
- implementation
priority: Medium
references:
- TASK-12905
- TASK-12164
- Docs/superpowers/specs/2026-07-07-character-expression-editor-onboarding-design.md
documentation:
- Docs/superpowers/specs/2026-07-07-character-expression-editor-onboarding-design.md
- Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md
- apps/packages/ui/src/components/Option/Characters/character-expression-images.ts
- apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts
- apps/packages/ui/src/components/Option/Characters/utils.ts
- apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx
- apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx
- apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx
- apps/packages/ui/src/components/Option/Characters/CharacterDialogs.tsx
- backlog/tasks/task-12906 - Implement-character-expression-editor-onboarding.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved shared character expression image editor, chat discovery nudge, and browser-extension handoff from the 2026-07-07 design spec. Work should stay in shared WebUI/extension UI paths and reuse existing character metadata and avatar image handling patterns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-07-07-character-expression-editor-onboarding-implementation-plan.md. Plan-document review loop completed. Reviewer iterations found and plan fixed: nudge dismissal server/user/character scoping, circular dependency between row helpers and character utils, and invalid raw extensions behavior for untouched empty starter rows. Final review status: Approved. Advisory starter-state load test was added to the plan.

Task 1 metadata helper contract completed in commit e5ec2d45cb. Added regression coverage for arbitrary safe expression state image keys, canonical mood image precedence over legacy aliases, canonical write cleanup, empty-map removal, and legacy alias resolution fallback. Verification: initial focused Vitest run failed on nested legacy alias cleanup and legacy resolver fallback; final `bunx vitest run src/utils/__tests__/character-mood.test.ts` from apps/packages/ui passed 11 tests. `git diff --check` passed. Bandit was attempted against touched TypeScript files with the project venv and produced no findings, with expected TypeScript parse errors because Bandit is Python-only.

Task 1 review fix: `upsertCharacterMoodImage` and `removeCharacterMoodImage` now normalize image-map keys with `normalizeCharacterEmoteState`, matching the arbitrary safe state contract. Added regression coverage for upserting/removing `smirk`. Red check failed because `smirk` upsert resolved to an empty string; final `bunx vitest run src/utils/__tests__/character-mood.test.ts` from apps/packages/ui passed 12 tests and `git diff --check` passed. Bandit was attempted against touched TypeScript files with the project venv and produced no findings, with expected TypeScript parse errors because Bandit is Python-only.

Task 2 pure editor row helpers completed. Added `character-expression-images.ts` with starter row loading, legacy metadata loading, row normalization errors for invalid/duplicate/incomplete custom rows, and canonical mood image map conversion. Wired `applyCharacterMetadataToExtensions()` and `buildCharacterPayload()` to merge expression rows through canonical `tldw.mood_images` only when rows or existing mood metadata require a write; invalid raw extensions are preserved for untouched empty starter rows and return null when expression metadata must be written. Verification: red check `bunx vitest run src/components/Option/Characters/__tests__/character-expression-images.test.ts` failed on missing helper module as expected. Final focused tests from `apps/packages/ui`: `bunx vitest run src/components/Option/Characters/__tests__/character-expression-images.test.ts src/utils/__tests__/character-mood.test.ts` passed 20 tests. `git diff --check` passed. Bandit was run with the shared repo venv at `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/packages/ui/src/components/Option/Characters/character-expression-images.ts apps/packages/ui/src/components/Option/Characters/utils.ts -f json -o /tmp/bandit_character_expression_images.json`; it produced no findings, with expected TypeScript AST parse errors because Bandit is Python-only. Direct UI package `bunx tsc --noEmit --pretty false --project tsconfig.json` first OOMed at the default Node heap, then completed with `NODE_OPTIONS=--max-old-space-size=8192` and failed on existing unrelated test type errors outside the touched files.

Task 3 Character Editor expression section completed. Added `CharacterExpressionImagesSection.tsx` as a compact Form.List section with starter/custom rows, row-level validation display, URL/upload/generate source modes, row thumbnails, preview picker, base-avatar fallback, and copyable `Emote: <state>` directive. Added focused component coverage for starter rows and custom add, copy action, duplicate/missing-image messages, preview fallback on empty/failed expression images, and URL mode row value updates. Red check from `apps/packages/ui`: `bunx vitest run src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx` failed on missing component import as expected. Final focused run passed 5 tests. `git diff --check` passed. Bandit was run with the shared repo venv at `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx -f json -o /tmp/bandit_character_expression_section.json`; it produced no findings, with the expected TypeScript AST parse error because Bandit is Python-only. Direct UI package `bunx tsc --noEmit --pretty false --project tsconfig.json` OOMed at the default Node heap; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false --project tsconfig.json` completed and failed on existing unrelated test type errors outside the touched files, with no errors reported for the new component.

Task 3 review fix completed. Hardened expression row image writes so upload/generate completions resolve the current Form.List row by stable `row.id` and no-op if the row no longer exists, preventing stale async completions from writing to the wrong index after row removal. Added visible copy feedback with `message.success` on successful clipboard writes and `message.error` when clipboard support is missing or `writeText` rejects. Added regression coverage for clipboard success, clipboard unavailable/rejected failures, and delayed upload completion after removing an earlier row. Verification from `apps/packages/ui`: `bunx vitest run src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx` passed 8 tests. `git diff --check` passed. Bandit was run with the shared repo venv at `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx -f json -o /tmp/bandit_character_expression_section_review_fix.json`; it produced no findings, with the expected TypeScript AST parse error because Bandit is Python-only.

Task 4 Character Editor save/load wiring completed. Replaced the legacy mood-images placeholder in the advanced Metadata section with `CharacterExpressionImagesSection`, seeded `expression_images` for create/edit/duplicate form state, preserved the hidden `starter` flag through Form.List submission, and added Ant Form submit validation that reuses `normalizeExpressionImageRows()` plus blocks invalid raw Extensions JSON when expression rows must be merged. Added manager coverage proving legacy `tldw.moodImages` starter/custom rows load into the edit form and save as canonical `extensions.tldw.mood_images` with legacy aliases removed. Red checks: `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "expression"` failed because the expression URL field was missing; `bunx vitest run src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx -t "blocks expression"` failed because submit validation did not participate in Ant Form. Final verification from `apps/packages/ui`: requested `bunx vitest run src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx src/components/Option/Characters/__tests__/character-expression-images.test.ts src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "expression|advanced"` passed 15 tests across 3 files. Focused manager expression run also passed. `git diff --check` passed. Bandit with the shared repo venv wrote `/tmp/bandit_character_editor_save_load.json` with `results=[]` and expected TypeScript parse errors. `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false --project tsconfig.json` completed and failed on existing unrelated test type errors outside the touched character files.

Task 4 spec-review fixes completed. Diagnosed the required filtered Vitest failure as a cold lazy-load timing issue in the create-mode manager test: Task 4 made `CharacterEditorForm` import the expression editor, so the default one-second `waitFor` could expire before the lazy drawer form reached the submit button. Kept the behavior under test and extended only that wait to 15s, matching the existing edit helper timeout. Expanded the real editor integration fixture to include top-level `mood_images`, top-level `moodImages`, and nested legacy `tldw.moodImages`, while retaining the exact canonical payload assertion for `extensions.tldw.mood_images`. Verification from `apps/packages/ui`: focused manager run `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "promotes prompt preset|seeds expression"` passed 2 tests; required command `bunx vitest run src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx src/components/Option/Characters/__tests__/character-expression-images.test.ts src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "expression|advanced"` passed 15 tests across 3 files. `git diff --check` passed before commit.
Task 4 code-quality review fix completed. Draft restore paths in `CharacterDialogs.tsx` now seed `expression_images` from an explicit draft field when present, otherwise from `expressionRowsFromExtensions(draft.extensions)`, preserving legacy mood image metadata through real create/edit draft restore saves. Added manager regression coverage that restores a legacy `moodImages` draft without `expression_images`, verifies the restored expression row is present in the form, and saves canonical `extensions.tldw.mood_images` instead of deleting the image. Verification from `apps/packages/ui`: focused `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "restores draft expression"` passed; required `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "draft|expression|advanced"` passed 5 tests.
Follow-up verification for the Task 4 draft-restore fix: `git diff --check` passed. Bandit was run with the shared repo venv against `CharacterDialogs.tsx` and `Manager.first-use.test.tsx`; `/tmp/bandit_task4_draft_restore_fix.json` has `results=[]` with expected TypeScript AST parse errors.
Task 5 Chat Setup Nudge completed. Added a scoped, dismissible expression setup nudge in `PlaygroundComposerNotices` after `ChatFirstRunNudge`, using `getCharacterMoodImagesFromExtensions()` so canonical and legacy expression image metadata suppress the nudge. Dismissal keys include stable server, user, and character scope in order, with no persistent key when a selected character has no stable id/slug. `PlaygroundForm` now passes the canonical server URL as the available stable server scope. Added focused notice tests for nudge visibility, configured-image suppression, scoped localStorage dismissal, same character id on different server scopes, unscoped session-only dismissal, and the Characters route fallback link. Verification from `apps/packages/ui`: red `bunx vitest run src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx -t "expression"` failed as expected on missing nudge UI; green focused run passed 5 tests; full `bunx vitest run src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx` passed 10 tests. `git diff --check` passed. Bandit first failed because the worktree has no `.venv`; rerun with the shared repo venv wrote `/tmp/bandit_task5_chat_setup_nudge.json` with `results=[]` and expected TypeScript AST parse errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
