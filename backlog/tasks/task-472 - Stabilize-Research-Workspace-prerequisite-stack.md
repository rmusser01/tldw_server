---
id: TASK-472
title: Stabilize Research Workspace prerequisite stack
status: In Progress
labels:
- research-workspace
- packaging
- webui
- backend
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
- https://github.com/rmusser01/tldw_server/pull/2018
- Docs/superpowers/plans/2026-05-23-research-workspace-prerequisite-stack-packaging-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Package the existing Research Workspace WIP from the dirty checkout into a clean reviewable branch based on origin/main. Scope includes the hard /research-workspace route replacement, legacy storage inventory gate, server bootstrap/trust panel wiring, migration protocol API, and related docs/tests without unrelated chat/sidebar/writing changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Clean branch is based on `origin/main` and contains only the Research Workspace prerequisite stack, with chat-workspace, prototype-workspace, and writing/playground changes excluded.
- [ ] Active WebUI and extension routes use `/research-workspace` and do not register, alias, or redirect `/workspace-playground`.
- [ ] Legacy local storage is inventoried and gated so unknown or unmapped data is never deletion-eligible.
- [ ] Backend migration protocol and source/status/capability APIs are included with focused pytest coverage and Bandit verification for touched Python.
- [ ] Frontend telemetry, prefill, trust panel, route, tutorial, and legacy inventory behavior have focused Vitest/e2e coverage.
- [ ] Real backend plus CDP browser smoke validates `/research-workspace`, old-route 404 behavior, and backend status/capability calls where feasible.
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
