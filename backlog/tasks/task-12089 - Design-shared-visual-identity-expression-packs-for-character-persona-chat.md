---
id: TASK-12089
title: Design shared visual identity expression packs for character/persona chat
status: In Progress
modified_files:
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for shared Visual Identity Packs that support SillyTavern-style expression ZIP imports, animated raster expression assets, character/persona defaults, chat portrait and stage rendering, and future VN role binding compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and revised after review for shared Visual Identity Packs. Latest revision clarifies actor-scoped default resolution when both character and persona packs exist, separates pack containers from mutable drafts and immutable versions, reserves generated source type for future workflows, collapses manual picker and /emote into one session override priority, adds retention/tombstone rules for replayed assets, and adds a V1 expression baseline appendix. Verification: self-review scan found no unresolved placeholder markers; design-only change so Bandit is not applicable.
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
