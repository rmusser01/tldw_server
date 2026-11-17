---
id: TASK-13647
title: Qualify quota and expected-user fixtures against current contracts
status: Done
labels:
- chat
- ci
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair five reproduced CI failures in existing quota and identity tests without changing production enforcement. PR #3229.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Original failure logging and quota assertions retained with current service shape
- [x] #2 Migration 100 skip behavior and current head metadata both verified
- [x] #3 Exact expected-user route coverage retains guard as first dependency
- [x] #4 Focused suite, lint, security validation and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: 5 failed, 22 passed, 21.00s on four original suites. ADR assessment: no new ADR required; existing quota/identity rules are unchanged. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_auth_fixtures_13647.md
Reviewed GREEN:27passed0skip167warnings20.97s process0. Ruff/Black/diffcheck0. Bandit0new10unchanged synthetic test literal findings/errors0 vsparent. Independent reviewer requested ledger-existence and distinct authoritative-fallback assertions, both implemented and re-reviewed clear. Review:Docs/Reviews/CHAT_AUTH_FIXTURE_CONTRACTS_13647_2026_10_13.md. No production changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Five reproduced fixture failures repaired without production changes;27connectedtests pass, review and exact hooks clear, static checks pass,0newBanditfindings. PR#3229.
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
