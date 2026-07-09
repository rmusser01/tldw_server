---
id: TASK-12170
title: Add planned flashcard generation types and counts
status: In Progress
modified_files:
- tldw_Server_API/app/api/v1/schemas/flashcards.py
- tldw_Server_API/app/api/v1/endpoints/flashcards.py
- tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py
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
