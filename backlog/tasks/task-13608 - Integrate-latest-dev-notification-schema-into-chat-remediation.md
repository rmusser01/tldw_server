---
id: TASK-13608
title: Integrate latest dev notification schema into chat remediation
status: Done
labels:
- chat
- integration
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate origin/dev 2a8ffa78a839a53171a27bc1c5f3ce811bb892b3 into the isolated chat remediation branch and qualify the affected notification consumer, canonical API contract, and retention lease boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Latest dev is included without changing upstream behavior or staging unrelated files.
- [x] #2 Affected notification, chat retention, and canonical API checks pass in the supported environment with skips recorded.
- [x] #3 Integration and independent review evidence are recorded while broad chat UX acceptance remains open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable decision; governing ADR-069 canonical API and ADR-049/050 plus proposed ADR-071 remain applicable. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_latest_dev_13608.md.
Merge sole conflict resolved by canonical regeneration.49pass24officialPGskip247warnings12.95s. Full fingerprint/path checks2129/3301 SHA256 bd7997199762d42481de4c4b32536f3c82f279d695c86ff44afa427b36771212. Ignored rich schema regenerated Node26 all10 kinds. Bandit actual3prod0/errors0; Ruff4files7/7 exactbaseline; initial normalhooks0. Evidence Docs/Reviews/CHAT_LATEST_DEV_13608_2026_10_12.md. Independent reviews pending final verdict; broad acceptance open.
Both final independent source reviews bounded CLEAR. Incoming consumer test validates persisted kind, not five-kind HTTP producer-to-reader roundtrip; existing inbox/API schema checks additionally run. All stages complete; only own plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_latest_dev_13608_completed.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Latest dev2a8ffa78 integrated locally with upstream Python files preserved. Sole generated fingerprint conflict regenerated from combined2129/3301 contract; paths check pass and ignored richtypes include10 notification kinds.49pass24officialPGskip; actual3prodBandit0/errors0; Ruff7/7 exactbaseline; independent reviews bounded CLEAR. No runtime/profile/browser/publicH2 changes; broader acceptance remains open.
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
