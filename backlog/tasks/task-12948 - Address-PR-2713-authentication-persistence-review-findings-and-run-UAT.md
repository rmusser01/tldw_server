---
id: TASK-12948
title: Address PR 2713 authentication persistence review findings and run UAT
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 05:01'
labels: []
dependencies: []
references:
  - TASK-12106
  - TASK-12108
  - 'https://github.com/rmusser01/tldw_server/pull/2713'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-11-pr-2713-auth-persistence-review-remediation-design.md
  - >-
    docs/superpowers/plans/2026-07-11-pr-2713-auth-persistence-review-remediation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve every Critical, Important, and Minor finding from the dedicated review of PR #2713. Centralize effective auth resolution across WebUI and extension transports, make cookie logout idempotent and no-store, harden secret preservation/clearing, add request-level lifecycle coverage, and complete browser UAT for WebUI and the loaded extension.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All HTTP, background, and upload transports use validated effective cookie-session or origin-bound device/session credentials.
- [ ] #2 Cookie-session logout revokes active sessions, clears stale or invalid cookies idempotently, clears client markers, and returns no-store responses.
- [ ] #3 Quickstart scrubbing preserves manual session secrets only when complete active connection metadata matches.
- [ ] #4 Credential clearing reports failure unless persistent and session secrets are both cleared.
- [ ] #5 Lifecycle tests authenticate through real production request paths after reload and relaunch and prove session expiry after browser restart.
- [ ] #6 A required CI gate runs the lifecycle regression suites.
- [ ] #7 Full WebUI and loaded-extension UAT passes for device, session, cookie, logout, reload, and relaunch flows.
- [ ] #8 Focused tests, lint, type, build, Bandit, and diff verification pass or only documented unrelated baselines remain.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: docs/superpowers/plans/2026-07-11-pr-2713-auth-persistence-review-remediation-plan.md. Execute Tasks 1-6 with TDD: shared effective-auth resolver; WebUI/extension transport integration; idempotent logout and truthful clearing; bootstrap scrub hardening; authenticated lifecycle CI; full verification and UAT.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
