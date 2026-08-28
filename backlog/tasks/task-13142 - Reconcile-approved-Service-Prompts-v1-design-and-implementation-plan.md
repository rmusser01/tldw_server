---
id: TASK-13142
title: Reconcile approved Service Prompts v1 design and implementation plan
status: Done
updated_date: '2026-08-28 05:24'
labels:
- service-prompts
- planning
priority: high
references:
- TASK-12955
- TASK-12956
- TASK-12958
- commit:1a038599753e780f32f62243871026ca9b6d2c06
documentation:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
- Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the approved four-prompt Service Prompts v1 specification onto current dev, retire the superseded broad rollout artifacts, and produce a lean dependency-ordered TDD implementation plan against the current backend, WebUI, and browser extension.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved four-prompt v1 specification is present on current dev history with approval/provenance recorded.
- [x] #2 Superseded broad-rollout Service Prompts plans and To Do tasks are archived or removed without touching unrelated Research Discovery work.
- [x] #3 A current-code implementation plan gives exact files, TDD steps, verification commands, security checks, and small commit boundaries.
- [x] #4 The required plan-document reviewer reports no material issues.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile the approved specification and superseded artifacts. 2. Map current backend/WebUI/extension seams. 3. Write the lean TDD implementation plan. 4. Run the plan-review loop. 5. Verify and commit the planning artifacts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reconciled onto origin/dev 4c2ad2070e. Ported the approved specification from 1a038599753e780f32f62243871026ca9b6d2c06 and committed the reconciliation as e6665ddf89; archived superseded broad-rollout tasks and removed obsolete plans/validator without touching Research Discovery work. Wrote the current-code TDD plan at Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md. Self-review verified every Modify/Delete path exists, every Create parent exists, 100 Markdown code fences are balanced, and git diff --check passes. The independent reviewer initially found three material gaps (independent mode parameter carriers, all-four-definition runtime E2E/reset proof, and uncommitted Backlog metadata); all three were corrected, and the same reviewer then approved the complete plan with no issues or recommendations. No runtime code was changed. Bandit was not run because this task changes only planning, documentation, and Backlog records. CI shard work was deliberately omitted per the requester; the implementation plan keeps future Python tests in already-covered directories.

Backlog ID reconciliation: the branch-local planning ID TASK-12973 first moved to legacy TASK-13013 after colliding with dev Embeddings work. TASK-13013.10 later moved this completed record to canonical TASK-13142 so the public release-readiness program uniquely owns TASK-13013. Historical commit subjects retain their immutable IDs; current planning and implementation links use TASK-13142.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled the approved lean four-prompt Service Prompts v1 design onto current dev, retired superseded broad-rollout planning artifacts, and produced an independently approved, dependency-ordered TDD implementation plan covering the backend registry/store/API, shared WebUI-extension Settings editor, legacy migration, immutable request snapshots, all named runtime consumers, cross-host behavior, security checks, and clean Backlog/PR handoff. The plan explicitly excludes approval/history/deployment/Jobs/second-database machinery and CI shard edits.
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
