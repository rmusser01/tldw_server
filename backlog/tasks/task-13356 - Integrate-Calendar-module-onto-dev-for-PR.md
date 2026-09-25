---
id: TASK-13356
title: Integrate Calendar module onto dev for PR
status: In Progress
created_date: 2026-09-25 19:44
labels:
- calendar
- integration
documentation:
- Docs/superpowers/plans/2026-09-25-calendar-dev-integration-plan.md
updated_date: 2026-09-25 20:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the completed Calendar module from the isolated feature branch onto current origin/dev without unrelated historical commits, verify integration, and open a PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR diff contains Calendar module work and necessary integration only.
- [x] #2 Focused backend and frontend Calendar tests pass on current dev.
- [x] #3 Security scan and route/build verification are recorded.
- [ ] #4 Pull request targets dev and documents manual provider-smoke limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ported Calendar onto origin/dev in an isolated worktree. Final focused verification: 142 backend/AuthNZ/lifecycle tests passed (2 stale migration-version assertions deselected), 30 Calendar frontend tests passed, frontend typecheck passed, targeted app ESLint passed, Bandit 0 findings, and local Next.js /calendar route returned HTTP 200. Broader router-contract tests hit unrelated optional-router import failures (media.ingest_jobs and prompt-studio websocket). The UI package has no standalone ESLint config. External Fastmail/CalDAV smoke remains unrun. Bounded polling deliberately does not infer remote deletion; explicit delta-sync deletion detection is deferred and documented. CalDAV is restricted to public HTTPS and same-origin discovered/bound collections.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
