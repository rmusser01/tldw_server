---
id: TASK-12948
title: Address PR 2713 authentication persistence review findings and run UAT
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-12 14:03'
labels: []
dependencies: []
references:
  - TASK-12106
  - TASK-12108
  - 'https://github.com/rmusser01/tldw_server/pull/2713'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-11-pr-2713-auth-persistence-review-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve every Critical, Important, and Minor finding from the dedicated review of PR #2713. Centralize effective auth resolution across WebUI and extension transports, make cookie logout idempotent and no-store, harden secret preservation/clearing, add request-level lifecycle coverage, and complete browser UAT for WebUI and the loaded extension.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All HTTP, background, and upload transports use validated effective cookie-session or origin-bound device/session credentials.
- [x] #2 Cookie-session logout revokes active sessions, clears stale or invalid cookies idempotently, clears client markers, and returns no-store responses.
- [x] #3 Quickstart scrubbing preserves manual session secrets only when complete active connection metadata matches.
- [x] #4 Credential clearing reports failure unless persistent and session secrets are both cleared.
- [x] #5 Lifecycle tests authenticate through real production request paths after reload and relaunch and prove session expiry after browser restart.
- [x] #6 A required CI gate runs the lifecycle regression suites.
- [x] #7 Full WebUI and loaded-extension UAT passes for device, session, cookie, logout, reload, and relaunch flows.
- [x] #8 Focused tests, lint, type, build, Bandit, and diff verification pass or only documented unrelated baselines remain.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed via staged TDD, independent spec and quality reviews, final whole-PR review, rebase, and real-Chromium UAT. The temporary implementation plan was removed after completion per repository policy; commit history preserves its execution record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Final base: origin/dev dc25171a8335ba1e0a4924788847eb1fb197b1c8. Final review fix: 9ef8babdb1; whole-PR re-review APPROVED. Backend/AuthNZ/workflow matrix: 119 passed. Explicit changed frontend/shared matrix: 40 files and 560 tests passed. Final UI interaction delta: 2 files and 9 tests passed. Production Chrome extension build passed. Full frontend lint passed with 0 errors and 174 existing warnings. Bandit medium/high passed across all changed Python production modules. git diff --check passed. Standalone frontend typecheck reports only the unrelated unchanged QuickIngestWizardModal.tsx:1813 baseline. Real-Chromium UAT after all fixes: WebUI device/session lifecycle 2 passed; loaded production extension device/session lifecycle 2 passed; same-origin HttpOnly cookie lifecycle 1 passed. Cookie UAT covers bootstrap, unsafe CSRF mutation without API-key headers, production logout, manual fallback, stale/no-cookie idempotency, rotation, and secret-free WebSockets. The dependency-aware changed-only Vitest selector left a stale tool session after unrelated large suites, so it was terminated and replaced by the explicit deterministic 40-file matrix. No functional blockers remain.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed every review finding for PR 2713. Same-origin single-user WebUI auth now uses persistent HttpOnly sessions without exposing the API key; remote manual connections support explicit device/session persistence; all transports resolve origin-bound effective auth; logout is strict, idempotent, no-store, and fail-safe on backend uncertainty; secret clearing and bootstrap scrubbing are truthful and canonical; onboarding and Settings recover correctly for cookie-only state; production lifecycle suites are required in frontend CI. Final WebUI, loaded-extension, and cookie UAT all pass on the rebased branch.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
