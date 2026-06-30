---
id: TASK-575
title: Stabilize Research Workspace prerequisite stack
status: Done
labels:
- research-workspace
- packaging
- webui
- backend
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
- https://github.com/rmusser01/tldw_server/pull/2018
- https://github.com/rmusser01/tldw_server/pull/2187
- Docs/superpowers/plans/2026-05-23-research-workspace-prerequisite-stack-packaging-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Historical tracker for packaging the Research Workspace prerequisite stack out of the dirty checkout into a clean reviewable branch. The implementation shipped through PR #2018 and later Research Workspace follow-up tasks; this record is renumbered from the duplicated TASK-472 to TASK-575 so the Research Workspace record has an unambiguous identifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Clean branch was packaged as a focused Research Workspace prerequisite stack, with unrelated chat-workspace, prototype-workspace, and writing/playground changes excluded from the reviewable PR.
- [x] Active WebUI and extension routes use `/research-workspace` and do not register, alias, or redirect `/workspace-playground`.
- [x] Legacy local storage is inventoried and gated so unknown or unmapped data is never deletion-eligible.
- [x] Backend migration protocol and source/status/capability APIs landed with focused pytest coverage and Bandit verification for touched Python.
- [x] Frontend telemetry, prefill, route, tutorial, and legacy inventory behavior landed with focused Vitest/e2e coverage; the earlier trust-panel surface was superseded by later Research Workspace UX corrections.
- [x] Real backend plus CDP browser smoke validated `/research-workspace`, old-route 404 behavior, and backend status/capability calls during the merged Research Workspace validation sequence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Renumbered from `TASK-472` to `TASK-575` after PR #2187 exposed that the old identifier still collided with unrelated Character Chat, Watchlists, and Persona task records.
- The original prerequisite packaging work is historical and complete; current Research Workspace implementation, source status drilldown, migration protocol, no-redirect route replacement, and UAT follow-ups are tracked by the later `TASK-463.*`, `TASK-469`, `TASK-478.*`, and Deep Research import tasks.
- Backlog/MCP lookup should now treat `TASK-575` as the canonical Research Workspace prerequisite stack record.
- Verification for this closeout is backlog-only: duplicate-ID/reference scans and `git diff --check`. Bandit is not applicable because no Python/runtime code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the stale Research Workspace prerequisite stack tracker as a completed historical record and renumbered it from the duplicated `TASK-472` to `TASK-575`. The associated implementation already shipped through PR #2018 and subsequent Research Workspace hardening/UAT tasks; this closeout removes the Research Workspace record from the remaining unrelated `TASK-472` collisions.
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
