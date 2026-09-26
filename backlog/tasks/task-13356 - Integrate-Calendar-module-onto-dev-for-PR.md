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
updated_date: 2026-09-26 04:39
references:
- https://github.com/rmusser01/tldw_server/pull/3019
modified_files:
- .github/workflows/ci.yml
- apps/packages/ui/src/routes/__tests__/calendar-route.test.ts
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
- [x] #4 Pull request targets dev and documents manual provider-smoke limits.
- [x] #5 Calendar tests are assigned to the Python 3.12 and 3.13 full-suite shards and the shard coverage guard passes.
- [ ] #6 PR #3019 is rebased onto latest dev, actionable Qodo feedback is addressed with verification, and merge status or remaining gates are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved CI follow-up: reproduce the shard coverage failure; add Calendar to gap-verified-2 in the Python 3.12 and 3.13 matrices; run the guard, focused Calendar tests and workflow validation; commit and push to PR #3019.
User-authorized follow-up: fetch latest dev and PR head; rebase the isolated Calendar branch without losing remote changes; obtain and verify Qodo review feedback; implement regression-tested fixes; run focused backend/frontend tests, Bandit and required CI; merge only after all required review and repository policy gates are met.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ported Calendar onto origin/dev in an isolated worktree. Final focused verification: 142 backend/AuthNZ/lifecycle tests passed (2 stale migration-version assertions deselected), 30 Calendar frontend tests passed, frontend typecheck passed, targeted app ESLint passed, Bandit 0 findings, and local Next.js /calendar route returned HTTP 200. Broader router-contract tests hit unrelated optional-router import failures (media.ingest_jobs and prompt-studio websocket). The UI package has no standalone ESLint config. External Fastmail/CalDAV smoke remains unrun. Bounded polling deliberately does not infer remote deletion; explicit delta-sync deletion detection is deferred and documented. CalDAV is restricted to public HTTPS and same-origin discovered/bound collections.
PR #3019 CI failed Shard coverage guard because 10 Calendar test files are unassigned. Reproduced locally: shards=811, test_files=4802, new_uncovered=10, exit=1. User approved the scoped CI fix. Backlog MCP/CLI global searches stalled on active branch scanning; searched local task records and retained existing TASK-13356.
CI follow-up verification: added Calendar to gap-verified-2 in both Linux Python 3.12/3.13 matrices. Guard passed (shards=812, test_files=4802, new_uncovered=0); direct YAML assertions confirmed both assignments; 112 Calendar tests passed on local Python 3.11; workflow-contract suite had 56 passed and 1 pre-existing chunking-coverage assertion failure at line 1511, reproduced against HEAD workflow identical to origin/dev. Pre-commit checks passed. Bandit is not applicable to this YAML/task-only change; no Python source changed. Initial hook run hit disk exhaustion; removed only this Calendar worktree generated .next cache (1.1 GB), with no preview listener on port 13008.
User requested rebase onto latest dev, address all Qodo issues/comments once posted, then merge. Initial PR state: draft, BEHIND, head c8295ff2b155ae5db01b75c777e51b50ddb1c1b8. No PR reviews or inline review comments have been posted yet.
Rebased cleanly onto origin/dev 59bd584503. Range-diff confirms the three original Calendar implementation/docs/CI commits are unchanged. Frontend verification from apps/tldw-frontend reproduced 2 route-test failures caused by process.cwd assumptions; replaced them with the existing fileURLToPath(import.meta.url) pattern. Red/green: frontend runner changed from 28 passed/2 failed to 30 passed, and shared UI runner route tests also passed (3). Backend Calendar suite: 112 passed; shard guard: new_uncovered=0; TypeScript: passed; Bandit on all touched backend source: 0 findings, 0 errors. pnpm exec attempted auto-install and could not find @tldw/ui; used installed Vitest/TypeScript binaries directly without tracked dependency changes. Qodo review remains unposted before publishing the rebased head.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opened draft PR #3019 against dev from codex/calendar-dev-pr. The PR ports local Calendar CRUD, agenda/week UI, linked scheduled-task views, and read-only CalDAV VEVENT import with opt-in Jobs. Verification: 142 focused backend/AuthNZ/lifecycle tests and 30 frontend tests passed; frontend typecheck and targeted app ESLint passed; Bandit reported 0 findings; local /calendar route returned HTTP 200. Two stale AuthNZ migration-version assertions and unrelated optional-router import failures are documented in the PR. Real-provider smoke and human-authored Change summary are required before merge. Remote deletion inference is deferred until complete delta sync.
Approved CI follow-up assigns all Calendar tests to the existing gap-verified-2 Python 3.12 and 3.13 full-suite shards without changing runtime behavior or exclusions. Local shard coverage now passes with zero uncovered files; 112 Calendar tests and all applicable pre-commit checks passed. Broader workflow-contract verification: 56 passed, 1 unrelated baseline chunking-coverage assertion failure, reproduced with the unchanged dev workflow. Hosted CI will validate the pushed commit.
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
