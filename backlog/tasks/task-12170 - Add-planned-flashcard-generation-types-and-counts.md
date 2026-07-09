---
id: TASK-12170
title: Add planned flashcard generation types and counts
status: In Progress
modified_files:
- tldw_Server_API/app/api/v1/schemas/flashcards.py
- tldw_Server_API/app/api/v1/endpoints/flashcards.py
- tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py
- tldw_Server_API/app/core/Workflows/adapters/content/_config.py
- tldw_Server_API/app/core/Workflows/adapters/content/generation.py
- tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py
- apps/packages/ui/src/services/flashcards.ts
- apps/packages/ui/src/services/__tests__/flashcards.test.ts
- apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/shared.ts
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/__tests__/shared.test.ts
- Docs/superpowers/plans/2026-07-08-planned-flashcard-generation-controls-implementation-plan.md
- backlog/tasks/task-12170 - Add-planned-flashcard-generation-types-and-counts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a small flashcard generation plan so users can request exact counts of supported card styles from one source selection. Reuse existing flashcard deck, generation, assets, and spaced-repetition paths.

Acceptance criteria:
- Flashcard generation accepts per-type counts for basic, reverse/basic_reverse, cloze/fill-in-the-blank, and true/false-style cards.
- Existing single card_type and num_cards request shape remains backward compatible.
- Generated true/false cards map cleanly onto existing flashcard storage without a new scheduler model.
- WebUI and extension generation entry points can request/preview the selected mix where applicable.
- Tests cover mixed flashcard generation requests and backward-compatible requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-08-planned-flashcard-generation-controls-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec: Docs/superpowers/specs/2026-07-08-planned-flashcard-generation-controls-design.md.
Implementation plan: Docs/superpowers/plans/2026-07-08-planned-flashcard-generation-controls-implementation-plan.md.

Approved design choices:
- Add optional `card_plan` to flashcard generation while preserving legacy `num_cards` plus `card_type`.
- Keep `/flashcards` simple mode as default and expose planned generation behind an Advanced mix toggle.
- Use response-only `generation_type` for preview labels and planned-count validation.
- Store generated true/false cards as existing `basic` flashcards; do not add scheduler or storage models.
- Do not silently change defaults in sidepanel, quiz companion, or Research Workspace callers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 1 backend schema contract slice: added card_plan request validation, response-only generation_type schema, focused mixed-plan/invalid-plan endpoint tests, and minimal generate endpoint passthrough/preservation needed for the contract test. Follow-up review regression added coverage for legacy default generate payloads omitting num_cards/card_type/card_plan. Verification: focused generate pytest passed (13 passed, 154 deselected); Bandit on touched backend schema/endpoint files reported 0 findings.

Task 2 endpoint validation slice: added planned-output rejection tests for count mismatches and missing/invalid generation_type, added test-mode planned fallback coverage, normalized adapter/test-mode output through one endpoint helper, preserved generation_type, mapped true_false to stored basic model_type, and validated exact planned counts before returning previews. Verification: focused generate pytest passed (17 passed, 154 deselected); Bandit on touched production endpoint reported 0 findings; touched-file Bandit passed with pytest assert rule B101 skipped.

Task 2 follow-up review fix: preserved planned-generation junk-row skip behavior by checking usable front/back before requiring generation_type, and added a regression where one junk row plus one valid planned basic row returns one card. Verification: red focused generate pytest failed before the endpoint fix; focused generate pytest passed after the fix (18 passed, 154 deselected); Bandit on the production endpoint passed; git diff --check passed.

Task 3 workflow adapter slice: added adapter config support for legacy card_type plus optional card_plan, planned prompt instructions with exact per-type counts and required generation_type, output preservation for generation_type, and true_false storage normalization to basic model_type. Verification: red focused adapter test failed before implementation; focused planned adapter test passed after implementation; FlashcardGenerateAdapter pytest passed (5 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 3 follow-up review fix: invalid direct workflow card_plan configs now return invalid_card_plan before LLM calls, and parsed adapter output is cleaned to dict flashcards before returning count. Verification: red focused adapter tests failed before implementation; focused regressions passed (7 passed, 116 deselected); FlashcardGenerateAdapter pytest passed (12 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 3 spec re-review fix: direct workflow card_plan validation now rejects duplicate card_type rows and rows with fields other than card_type/count before LLM calls. Verification: red focused invalid-plan adapter test failed on duplicate/extra-field cases before implementation; focused invalid-plan pytest passed (8 passed, 117 deselected); FlashcardGenerateAdapter pytest passed (14 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 3 strict-count follow-up: direct workflow card_plan counts now require actual positive int values, rejecting floats, bools, and numeric strings before LLM calls. Verification: red focused invalid-plan adapter test failed on coerced count cases before implementation; focused invalid-plan pytest passed (11 passed, 117 deselected); FlashcardGenerateAdapter pytest passed (17 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 3 raw-config follow-up: invalid raw num_cards now returns invalid_num_cards before LLM calls, focus_topics are normalized to stripped strings, and planned cards missing/invalid generation_type are skipped from direct adapter results. Verification: red focused adapter tests failed before implementation; focused regressions passed (4 passed, 128 deselected); FlashcardGenerateAdapter pytest passed (21 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 3 strict-num-cards follow-up: raw workflow num_cards now requires an actual positive int, preserving the omitted default while rejecting bools, floats, and numeric strings before LLM calls. Verification: red focused invalid-num-cards test failed on coerced values before implementation; focused invalid-num-cards pytest passed (5 passed, 130 deselected); FlashcardGenerateAdapter pytest passed (24 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 3 final direct-workflow follow-up: planned adapter output now validates cleaned generation_type counts against card_plan and returns card_plan_mismatch instead of wrong planned mixes. Verification: focused planned mismatch regressions passed (2 passed, 134 deselected); FlashcardGenerateAdapter pytest passed (25 passed, 111 deselected); Bandit on touched production adapter files reported 0 findings; git diff --check passed.

Task 4 frontend service/draft plumbing slice: added shared UI generation-plan types, typed optional card_plan requests, response-only generation_type metadata, hook cardPlan passthrough, generated-draft normalization that preserves true_false generation_type while keeping model_type storage-compatible, and focused service request-body coverage. Initial red test command could not load Vitest config because frontend dependencies were missing; after bun install, verification passed: bunx vitest run ../packages/ui/src/services/__tests__/flashcards.test.ts (9 passed), bun run typecheck, and git diff --check. Bandit skipped because this slice only touched frontend TypeScript and Backlog markdown.

Task 4 quality-review follow-up: generated draft normalization now falls back to normalized model_type when generation_type is absent while preserving explicit true_false, and the generation hook omits legacy card_type whenever cardPlan is provided. Added focused normalization regression coverage. Verification: red shared normalization Vitest failed before the fix; after the fix, shared normalization Vitest passed (1 passed), service Vitest passed (9 passed), bun run typecheck passed, and git diff --check passed. Bandit skipped because this follow-up only touched frontend TypeScript and Backlog markdown.
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
