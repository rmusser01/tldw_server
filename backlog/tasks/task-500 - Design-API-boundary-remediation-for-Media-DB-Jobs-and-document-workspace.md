---
id: TASK-500
title: Design API boundary remediation for Media DB Jobs and document workspace
status: Done
labels:
- design
- api
- backend
priority: Medium
documentation:
- Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md
modified_files:
- Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md
- backlog/tasks/task-500 - Design-API-boundary-remediation-for-Media-DB-Jobs-and-document-workspace.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for addressing the Codeslop Vibecheck findings across tldw_Server_API: consolidate Media DB update invariants, move Jobs event SQL behind JobManager public APIs, and move document workspace table ownership into Media DB repositories/migrations while preserving external HTTP API compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design captures the approved long-term stable internal boundary cleanup approach.
- [ ] #2 Spec covers architecture, components/data flow, error handling, compatibility, testing, rollout stages, and acceptance criteria.
- [ ] #3 Spec is linked from the Backlog task and committed with the task update.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written at Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md. Spec review subagent approved it with no blocking issues. Verification is documentation-only: no code tests or Bandit were run because this task only adds a design spec and Backlog tracking record. Follow-up implementation planning should keep the three rollout stages independently testable, lock identical-content media update behavior before coding Stage 1, and choose one post-commit hook ownership direction.
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
