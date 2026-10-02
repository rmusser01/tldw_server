---
id: TASK-13414
title: >-
  UX smoke hard-gate allowlist: 31 entries expired 2026-09-30, so UX Smoke Gate
  fails on every PR
status: To Do
assignee: []
created_date: '2026-10-01 17:54'
updated_date: '2026-10-01 17:55'
labels:
  - bug
  - webui
  - testing
  - ci
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
apps/tldw-frontend/e2e/smoke/smoke.setup.ts SMOKE_HARD_GATE_ALLOWLIST has 31 entries (owners WebUI and Platform) with expiresOn 2026-09-30. Since 2026-10-01 UTC the spec 'hard-gate allowlist entries have current ownership metadata' fails first, so UX Smoke Gate (frontend-ux-gates.yml; not a required check) is red on every PR. First seen on #3065 run 36801831063. The expiry is a review deadline, so blanket-extending the dates defeats it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each expired entry is re-checked against a current all-pages smoke run: removed if its noise no longer occurs, otherwise renewed with a new expiresOn and a linked task for the underlying noise
- [ ] #2 UX Smoke Gate passes the allowlist-metadata check
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
