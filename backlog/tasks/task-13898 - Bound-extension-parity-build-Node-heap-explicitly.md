---
id: TASK-13898
title: Bound extension parity build Node heap explicitly
status: In Progress
labels:
- ci
- extension
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up merged PR #2862: Extension Research Workspace Parity run 39610735435 job 119259527126 exhausted the default approximately 4 GiB V8 heap during wxt production build before parity tests. The user explicitly approved a scoped CI build-memory fix. Limit the setting to the Chrome build step; preserve tests, runner, timeouts, admission, and all other jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chrome extension parity build receives an explicit bounded heap allowance above the failing default.
- [x] #2 Workflow contract tests protect step-only scope and unchanged build/test commands.
- [x] #3 Validate workflow contracts, scoped security checks, and available production build without dependency installs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This adjusts a job-local resource allowance, not repository workflow policy, admission, permissions, runners, or application architecture. Plan: reproduce the missing heap contract; add only build-step NODE_OPTIONS; verify focused CI contracts and the production build with existing dependencies. Original OOM log: /private/tmp/pr2862-extension-parity-5bc.log. Parent task TASK-13134; PR #2862 merged as 5c2e2b4c804045aa01dc74e2854f53fb902a4849.
Investigation: build-with-profile.mjs forwards env unchanged to wxt via execFileSync, so no wrapper/application changes are needed. Existing WebUI parity memory allowance is 5120 MiB. Use that same bounded allowance only on the failing Chrome build step, leaving dependency installation and Playwright environments untouched.
Verification: new contract red before workflow change (KeyError env), then 112 workflow/admission tests passed on Python 3.12.13. Production Chrome build with NODE_OPTIONS=--max-old-space-size=5120 and minimal locales passed in 38.1s using existing dependency symlinks, Node22.20.0/macOS; no installs. CI Node20/Linux remains to be verified. Ruff check/format applied; Bandit test scope excluding normal test assertions B101: zero findings/errors. Independent read-only review clear; note V8 old-space cap is not total process RSS. No runner, timeout, permission, admission, browser test, or application changes. User explicitly authorized separate follow-up PR.
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
