---
id: TASK-13260
title: Run fresh single-user and multi-user UAT across core workflow loops
status: In Progress
created_date: 2026-09-15 01:27
labels:
- uat
- testing
- documentation
documentation:
- Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md
updated_date: 2026-09-15 01:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise fresh-data single-user and multi-user installations in the current checkout and walk through the user-defined core workflow loops A, B, and C. Keep a running tracker of bugs, failures, UX issues, workarounds, evidence, and coverage gaps. Awaiting the user's A/B/C definitions; clean dependency installation is not certified by reused environments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exercise single-user fresh setup and agreed core workflow loops A, B, and C.
- [ ] #2 Exercise multi-user fresh setup, admin and ordinary accounts, and agreed core workflow loops A, B, and C.
- [x] #3 Record all observed product issues with reproducible steps, expected and actual behavior, and evidence.
- [x] #4 Verify account isolation and document environmental limitations and untested scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Prepare isolated configurations, SQLite databases, ports, and browser sessions; exercise fresh setup and independent ingestion/chat/auth paths; obtain A/B/C definitions and walk through each; consolidate evidence and keep the running tracker current.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Checkpoint: 15 product findings recorded in Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md (five P1, six P2, four P3), with six environment/tooling observations. Fresh isolated single-user and multi-user SQLite state initialized using existing dependencies; no clean dependency/Docker/Postgres certification. Single-user real local chat and reload persistence pass; source ingestion and media search pass. Admin plus Alice/Bob authenticate, and API note/media isolation and ordinary-user admin denial pass. Multi-user onboarding skip fails CSRF; Alice's Notes UI is blocked by /api/v1/health/live requiring system.logs. Knowledge QA fails provider validation after optional RAG setup was deferred. User definitions for A/B/C remain outstanding, so neither loop coverage nor overall UAT is complete. Selected screenshots, snapshots, and API results retained under output/playwright/fresh-install-2026-09-14; private credentials/raw logs excluded. UAT services and synthetic state retained on ports 18000/18080 and 18001/18081 for continuation; original 8000 and model 9099 untouched. Backlog CLI collision replaced original untracked 13259; semantic content restored through MCP from task history, with original timestamp values documented in its recovery note. This UAT uses explicit 13260. No product code modified; Bandit/application tests are not applicable to the documentation/evidence checkpoint.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Initial UAT checkpoint only: fresh-data setup, real single-user chat, ingestion, authentication, and scoped API isolation exercised. Fifteen product findings and six environment observations logged with reproduction steps, workarounds, qualifications, and evidence. Awaiting A/B/C definitions and installation target confirmation before completing requested loop coverage. Task remains In Progress.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
