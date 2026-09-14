---
id: TASK-13137
title: Implement saved Notes graph views and layouts
status: To Do
created_date: 2026-08-27 02:21
labels:
- notes
- notes-graph
- webui
- extension
- second-brain
priority: Medium
dependencies:
- TASK-13138
updated_date: 2026-08-27 03:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users save and restore named Notes Graph workspaces, including focus, filters, visible relationship types, layout mode, viewport, and optional pinned node positions. Keep saved views owner-scoped and resilient to graph revision changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can create, rename, duplicate, update, list, load, and delete named graph views under the existing /api/v1/notes graph namespace.
- [ ] #2 A saved view stores the canonical dataset, focused note or scope, query/filter state, visible edge types, layout algorithm/options, viewport, and bounded pinned node coordinates without persisting transient graph responses.
- [ ] #3 Loading reconciles missing, trashed, restored, or newly related nodes safely, reports stale or unavailable focus targets, and never treats stored coordinates or graph IDs as authorization.
- [ ] #4 The shared WebUI/extension Graph mode exposes saved-view selection and explicit save/update commands with optimistic concurrency, unsaved-change indication, keyboard access, and responsive behavior.
- [ ] #5 Saved layouts are owner-scoped, RBAC-protected, versioned, exportable with Notes data, and compatible with SQLite/PostgreSQL and the established Sync policy or explicitly documented as local-only for the first slice.
- [ ] #6 CRUD, lifecycle reconciliation, security, payload limits, accessibility, responsive browser coverage, documentation, and Bandit verification are covered.
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
