---
id: TASK-406
title: Design Persona Visual post-setup management UX
status: Done
labels:
- persona
- visual-packs
- design
- webui
priority: medium
ordinal: 398
references:
- https://github.com/rmusser01/tldw_server/issues/1769
- https://github.com/rmusser01/tldw_server/issues/1510
documentation:
- Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md
modified_files:
- Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a Persona/Buddy design spec for the post-setup Persona Visual management experience in Persona Garden Visuals. Scope is documentation/tracking only: define how users understand and manage active packs, drafts, imports, exports, generated candidates, personal-library entries, duplicate-to-persona drafts, and recovery states without changing backend semantics or overlapping PR #1767 recipe-backed generation work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design defines post-setup active/draft/review/archived/failed pack management without changing backend semantics.
- [x] #2 Design identifies attention states for generated candidates, import/export jobs, invalid manifests, generation readiness, and stale library entries.
- [x] #3 Design preserves review-first and explicit activation semantics.
- [x] #4 Design explicitly excludes PR #1767 recipe-backed generation work and VN/CYOA scope.
- [x] #5 Design proposes small implementation slices and focused verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/specs/2026-05-16-persona-visual-post-setup-management-design.md`.
- Created GitHub tracking issue #1769 under the Persona/Buddy epic #1510.
- Verified with `git diff --check`.
- Bandit skipped because this slice only changes documentation and Backlog task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design completed for a post-setup Persona Visual management UX centered on the existing Persona Garden Visuals surface. The spec keeps the work parallel to PR #1767 and recommends a first implementation slice for a pure shared UI management-summary/attention model.
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
