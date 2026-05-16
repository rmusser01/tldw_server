---
id: TASK-399
title: Design staged bulk conference ingest workflow improvements
status: In Progress
labels:
- ux
- quick-ingest
- webui
- extension
priority: Medium
documentation:
- Docs/superpowers/specs/2026-05-16-bulk-conference-ingest-workflow-design.md
modified_files:
- Docs/superpowers/specs/2026-05-16-bulk-conference-ingest-workflow-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a critique-hardened staged design for PR-sized improvements that take Quick Ingest, WebUI, and extension support from the current conference playlist workflow to the ideal bulk conference ingest/review process.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design is grounded in current Quick Ingest, Media, Jobs, Knowledge QA, and extension surfaces.
- [ ] #2 Plan is split into PR-sized vertical slices with dependencies and acceptance criteria.
- [ ] #3 Known risks from the UX audit and critique pass are addressed explicitly.
- [ ] #4 No implementation code changes are made as part of this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted and repeatedly hardening-reviewed the staged bulk conference ingest workflow design. The final review pass clarified that planned collection items are created before job submission, completed media IDs resolve into those collection items, `submit_failed` carries metadata/error/export behavior consistently, PR 4/5 cover all-retry controls, and PR 8 owns selection-based retry. Awaiting user approval before implementation planning.
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
