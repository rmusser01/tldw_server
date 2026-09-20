---
id: TASK-13136
title: Implement library-wide recurring themes for Notes
status: To Do
created_date: 2026-08-27 02:21
labels:
- notes
- notes-graph
- themes
- second-brain
- ai
priority: Medium
dependencies:
- TASK-13138
updated_date: 2026-08-27 03:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Derive recurring themes across an owner's active Notes library and make them inspectable as source-grounded, refreshable graph concepts. Themes are derived summaries with note membership and evidence, not canonical tags or silently applied organization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bounded Jobs-backed analysis produces versioned theme snapshots for the active Notes dataset, with explicit model/provider provenance and refresh/staleness state.
- [ ] #2 Every theme includes a concise label and summary, member note IDs, source-grounded supporting excerpts, confidence/coverage metadata, and deterministic identifiers within a snapshot.
- [ ] #3 The Notes Graph API can return optional theme nodes and note-to-theme membership edges with hard caps, filters, truncation metadata, and drill-down support.
- [ ] #4 The Graph workspace provides a cluster overview and theme inspector that can refocus into member notes while preserving an accessible relationship-list equivalent.
- [ ] #5 Theme refreshes reconcile edits, trash/restore, deletions, merges/splits, model changes, empty libraries, and failed or cancelled runs without overwriting user tags.
- [ ] #6 RBAC, privacy boundaries, cost controls, observability, SQLite/PostgreSQL persistence, tests, documentation, and Bandit verification are covered.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
