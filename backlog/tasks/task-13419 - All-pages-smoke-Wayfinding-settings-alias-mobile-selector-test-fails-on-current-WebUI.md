---
id: TASK-13419
title: >-
  All-pages smoke: Wayfinding settings alias/mobile selector test fails on
  current WebUI
status: To Do
assignee: []
created_date: '2026-10-02 02:39'
labels:
  - webui
  - testing
dependencies: []
references:
  - apps/tldw-frontend/e2e/smoke/all-pages.spec.ts
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while re-running the full e2e/smoke/all-pages.spec.ts for TASK-13414 (2026-10-01, backend MINIMAL_TEST_APP single_user, WebUI standalone build:dev bundle and next dev --webpack). 'Smoke Tests - Wayfinding > settings alias and mobile section selector keep route context clear' fails deterministically on all retries in both runtimes. Standalone bundle: after /settings/image-gen redirects to /settings/image-generation, /settings at 390x844 has no level-1 heading matching /setup & recovery/i within 30s. next dev: the /settings/image-gen 'This route has moved' page never reaches /settings/image-generation within 30s. The UX Smoke Gate does not run this describe (it greps 'Smoke Tests - All Pages'), so CI stays green while the full spec is red. Not an allowlist issue: no console or request noise is involved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified: test drift versus a real redirect or mobile-heading regression on /settings
- [ ] #2 The Wayfinding settings alias test passes against the standalone bundle and next dev, or is updated to the current contract with a recorded reason
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
