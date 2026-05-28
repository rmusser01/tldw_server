---
id: TASK-525
title: Implement notes connection link clarity
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 00:02'
labels:
  - notes
  - webui
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR7 slice for the /notes UX remediation plan: make existing note relationships understandable and navigable without expanding the first-class link object model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manual note links, backlinks, graph edges, chat/message backlinks, source/clipper links, and reading-item links use understandable labels where those relationships already exist.
- [x] #2 Users can navigate from a note to a linked note target where route support exists.
- [x] #3 Missing, deleted, or inaccessible linked note targets show a clear unavailable state.
- [x] #4 Edge type labels distinguish manual links, backlinks, tags, and source membership instead of exposing raw internal labels.
- [x] #5 No new media, research, prompt, or other first-class link-target support is added in this slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-7-connections-and-cross-surface-link-clarity
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented frontend-only link clarity: readable graph edge labels, tag/source legend entries, relation-type badges, source-membership filtering, disabled unavailable linked-note targets, and unavailable-target filtering from manual link options. No new first-class media, research, prompt, or other link target support was added.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Notes connection/link clarity slice completed. Verification: 30 focused/adjacent Notes tests passed; extension TypeScript compile passed; git diff --check passed; local WebUI /notes smoke rendered the notes list/editor against 127.0.0.1:18001 with no console errors. Bandit skipped because only TypeScript/frontend and Backlog task files changed.
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
