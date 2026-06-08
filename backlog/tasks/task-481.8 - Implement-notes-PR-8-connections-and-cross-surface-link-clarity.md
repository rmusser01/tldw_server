---
id: TASK-481.8
title: Implement notes PR 8 connections and cross-surface link clarity
status: Done
labels:
- notes
- ux
- webui
- links
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 8 from the notes UX remediation plan: improve labels and unavailable states for existing note links, graph edges, chat/message backlinks, source/clipper links, and reading-item links without adding new first-class link targets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-8-connections-and-cross-surface-link-clarity
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR8 frontend-only connection clarity. Manual links, related notes, backlinks, source chips, and graph edges now use user-readable labels; missing/deleted linked notes render as unavailable instead of clickable placeholder notes; source labels normalize raw IDs while preserving human-readable graph labels. Focused connection tests passed for graph view, manual links, wikilinks, backlink labels, and source links (22 tests). Full Notes component sweep was also run: 66/67 files and 203/204 tests passed; the remaining deterministic failure is unrelated to PR8 in `NotesManagerPage.stage10.ai-title.test.tsx` (`LLM (quality)` strategy dropdown option not found). Browser smoke remains needs-verification because no live API/WebUI stack was started for this mocked-payload slice. Bandit was not applicable because no Python/backend files changed.
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
