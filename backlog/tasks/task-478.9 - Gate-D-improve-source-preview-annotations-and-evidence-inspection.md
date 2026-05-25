---
id: TASK-478.9
title: 'Gate D: improve source preview, annotations, and evidence inspection'
status: To Do
labels:
- research-workspace
- uat
- gate-d
- source-preview
- annotations
- citations
priority: Medium
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible gap: opening a source/annotation modal showed metadata and annotation fields but no useful ingested source-content preview. Annotation creation worked, but the user could not inspect what content was actually captured or citeable from that source.

User goal: inspect a source, verify captured text/snippets, add notes/annotations, and connect those notes to later evidence/citation workflows.

Scope:
- Add source-content preview for ingested text or extracted chunks with clear loading/error/empty states.
- Show citation/evidence snippets or chunk metadata when available, respecting readiness/status from TASK-478.3.
- Validate annotation create/edit/delete/read behavior and persistence if those actions exist or should exist.
- Ensure preview handles large sources with pagination, search-within-source, or bounded snippets rather than dumping unbounded content.
- Add tests for preview available, preview pending, extraction failed, large source, and annotation persistence paths.

Acceptance criteria:
- A user can open a workspace source and verify at least representative captured content or a precise reason content is unavailable.
- Annotation controls do not hide or replace source inspection.
- Evidence/citation snippets are linked to source identity and readiness where supported.
- Live CDP/Playwright validation covers preview and annotation behavior.

Depends on: TASK-478.3 for readiness semantics.
Parallelization: can run in parallel with acquisition/layout/onboarding once status fields are stable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
