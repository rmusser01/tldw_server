---
id: TASK-530.10
title: Implement Skills version-aware delete path
status: In Progress
labels:
- skills
- webui
- safe-operations
- backend
priority: high
ordinal: 530.1
parent_task_id: TASK-530
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.9 by adding version-aware single-skill delete behavior. Extend the frontend/API path so stale destructive deletes can be blocked and recovered before any bulk-delete work. Keep bulk delete, export feedback, and permission metadata panels out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Delete requests send an If-Match version when the frontend has a known skill version.
- [ ] #2 Backend delete validates If-Match consistently and returns a recoverable conflict for stale versions.
- [ ] #3 The Skills manager shows a clear reload-before-delete recovery message on stale delete conflicts.
- [ ] #4 Existing delete behavior remains compatible when no version is known.
- [ ] #5 Focused frontend and backend tests cover successful delete, no-version compatibility, and stale-version conflict handling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec approved by user and written at `Docs/superpowers/specs/2026-06-28-skills-version-aware-delete-design.md`. Design scope: expose row versions in Skills list summaries, send `If-Match` from frontend deletes when known, preserve unversioned delete compatibility, and show reload-before-delete recovery copy on stale conflicts.

Self-review completed before implementation planning. Found and patched a spec gap: `SkillSummary` is reused by `/skills/context` and async context integration paths, so requiring `version` must also update `_build_context_payload()` and context/MCP fixtures while keeping `context_text` unchanged. Also tightened delete conflict detection guidance to use a helper covering common wrapped error shapes.
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
