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
Create a design spec for addressing the Codeslop Vibecheck findings across tldw_Server_API: derive minimal-test router metadata from production RouterSpec definitions, consolidate Media DB update invariants, move Jobs event SQL behind JobManager public APIs, move document workspace table ownership into Media DB repositories/migrations, and move prototype promotion review decisions behind a public service method while preserving external HTTP API compatibility. Worker lifecycle state consolidation is explicitly out of scope for this series.
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
Design spec amended at Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md to include two additional verified findings in the remediation series: deriving minimal-test router metadata from production RouterSpec definitions, and moving prototype promotion review authorization/state transitions behind a public service method. The worker lifecycle state consolidation finding is explicitly documented as out of scope for this series. Local verification passed: git diff --check on the touched spec/task files produced no output, and the stale wording/placeholder rg scan returned no matches. No code tests or Bandit were run because this amendment only changes documentation and Backlog tracking. The subagent spec-review gate was not rerun in this turn because the current agent-delegation tool requires explicit user delegation permission.
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
