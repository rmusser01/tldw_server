---
id: TASK-13239
title: 'Run live UAT for reported admin login, profile, and chat scenarios'
status: In Progress
assignee: []
created_date: '2026-09-10 05:07'
updated_date: '2026-09-10 06:22'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2939'
documentation:
  - Docs/Reviews/ISSUES_2935_2938_LIVE_UAT_2026_09_10.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify PR #2939 against the original issue 2935–2938 reproduction scenarios using a real Node 20 admin UI, an isolated PostgreSQL-backed API provisioned by the repository fixture, browser login and settings paths, and live chat HTTP calls. Record evidence and any test-double limits; repair any defects exposed by the acceptance run and update the existing PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Node 20 admin UI accepts a real login JWT and rejects invalid tokens without HTTP 500
- [x] #2 Live PostgreSQL profile requests succeed with Bearer and API-key auth, populated sessions, and zero/one/multiple organization and team memberships
- [x] #3 Live chat requests resolve omitted and blank models from configuration, preserve explicit models, and fail clearly before dispatch without a model
- [ ] #4 Scenario results, environment, evidence, limitations, and any follow-up regression checks are recorded on PR #2939
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Match acceptance cases to original reports and prepare isolated live services. 2. Exercise Node 20 browser login plus live profile and chat requests, recording observed responses. 3. Investigate and repair failures, validate changes, retain evidence, and update the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Live PostgreSQL 18.6 UAT exposed an additional full-profile 500: Usage/audio_quota.py passes an ISO date string to asyncpg DATE in get_daily_minutes_used. Bearer and API-key calls both reproduce. Investigating matching backfill and jobs-started queries; add PostgreSQL regressions and repeat live checks after repair.

Live UAT after quota DATE fix: full profile Bearer and API key 200 for 0/1/2 org and team memberships; team override value/source correct; token refresh and subsequent profile200; real registration auto-creates org/team and profile200. Compiled Next16.2.2 on Node20.19.5 admin browser login and reload pass; invalid JWT matrix redirects307. Local provider wire logs verify streaming/nonstreaming omitted/blank/default/explicit/missing-default behavior; Ollama adapter file default also passes. WebUI Save reports server responded successfully; persistence check in progress.

Quota DATE fix verified: 4 new regressions red with asyncpg DataError, then 5 PostgreSQL +49 existing tests green; Ruff/Black/compilation/diff checks and scoped Bandit pass. Independent review found no actionable issues. Follow-up WebUI persistence retry exposed a separate event-loop/RBAC lock stall, tracked and being fixed under TASK-13240; final UAT report in Docs/Reviews/ISSUES_2935_2938_LIVE_UAT_2026_09_10.md.

Live acceptance verification is complete. Real browser admin login/reload and WebUI API-key Save/persistence pass; exact Node20.20.2 runtime also passes valid JWT and all five invalid/missing-token redirects. Six profile membership/authentication cases, session activity, refresh, real signup, and24 streaming/nonstreaming chat cases pass. UAT exposed and repaired audio DATE binding and RBAC event-loop stalls, with cache concurrency followups from review. Final fresh API and lock replays pass. Disposable DB removal and all owned server ports stopped are verified; temporary fixture and completed plan removed. Report and sanitized evidence are prepared for PR publication; full-suite/external-provider/Docker limits are explicit.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
