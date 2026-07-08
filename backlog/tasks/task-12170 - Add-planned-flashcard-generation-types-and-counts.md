---
id: TASK-12170
title: Add planned flashcard generation types and counts
status: To Do
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec: Docs/superpowers/specs/2026-07-08-planned-flashcard-generation-controls-design.md.

Approved design choices:
- Add optional `card_plan` to flashcard generation while preserving legacy `num_cards` plus `card_type`.
- Keep `/flashcards` simple mode as default and expose planned generation behind an Advanced mix toggle.
- Use response-only `generation_type` for preview labels and planned-count validation.
- Store generated true/false cards as existing `basic` flashcards; do not add scheduler or storage models.
- Do not silently change defaults in sidepanel, quiz companion, or Research Workspace callers.
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
