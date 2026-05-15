---
id: TASK-349
title: Design first-class Watchlist container and staged remediation plan
status: Done
assignee: []
created_date: '2026-05-15 01:11'
updated_date: '2026-05-15 01:14'
labels:
  - watchlists
  - ux
  - design
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an approved design specification for making Watchlist a first-class project-like container in the WebUI/API product model. The design must preserve existing Watchlists infrastructure while defining the user-facing container model, scoped child workflows, content-match alerts, defensible reports, extension-sized management expectations, and a staged implementation plan for follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec documents Watchlist as a first-class project-like container with intent, tracked scope, sources, monitors, alerts, review queue, reports, and lifecycle.
- [x] #2 Spec distinguishes content-match alerts from pipeline health issues and maps current run-stat alert rules and topic monitoring as dependencies rather than conflating them.
- [x] #3 Spec includes CTI/OSINT and news power-user workflow requirements with concrete domain fields and report expectations.
- [x] #4 Spec includes a staged remediation plan with stage goals, dependencies, complexity, tests, rollout gates, and risks/open questions.
- [x] #5 Spec is written under Docs/superpowers/specs/ and references the relevant existing watchlists docs/API/frontend/backend files.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created design spec at Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md. Verification: read the generated spec back with sed, ran git diff --check against the tracked path (no output), and ran awk trailing-whitespace check on the new spec (exit 0). Bandit skipped because this is a documentation-only design spec with no Python/code changes. Spec review was local because subagent delegation was not explicitly authorized in this session.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the approved first-class Watchlists design spec and staged remediation plan. The spec defines Watchlist as a project-like container, maps existing sources/jobs/runs/items/outputs/templates into scoped child workflows, separates content alerts from pipeline health issues, covers CTI/OSINT and news workflows, and lays out seven implementation stages with dependencies, tests, rollout gates, risks, and open questions.
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
