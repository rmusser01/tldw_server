---
id: TASK-13622
title: Restore authoritative quota read contracts in caller fixtures
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3229 quota-switch and deprecated-email characterization fixtures must declare authoritative storage reads for their synthetic users without weakening native usage or identity failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both original quota-switch and all caller characterization contracts pass with explicit quota read dependencies
- [x] #2 Unavailable native usage remains fail-closed with quotas on or off; exact synthetic-user quota lookup is asserted
- [x] #3 Bounded independent review, Python checks, Bandit and normal hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; fixture-only restoration of current authoritative storage service contracts. Related PR: https://github.com/rmusser01/tldw_server/pull/3229. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_quota_fixture_contracts_13622.md.
Reproduced four failures (27 passed): native ledger missing from an object-pool fixture and quota status lookup left live for synthetic user7. Declared scoped I/O dependencies without changing production code. Added four unavailable-native-usage controls and exact user7 quota lookup assertions. Final complete suites35/35; nearby quota resolver/profile error mapping24/24. Black and Ruff pass. Touched-test Bandit excludes intentional assertions (B101); two inherited static SQL matcher B608 findings match committed baseline, zero new findings. Independent bounded review is clear. Review: Docs/Reviews/CHAT_QUOTA_FIXTURE_CONTRACTS_13622_2026_10_12.md.
Normal exact-file pre-commit hooks passed. Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_quota_fixture_contracts_13622_completed.md. PR: https://github.com/rmusser01/tldw_server/pull/3229.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Declared authoritative native usage for quota-switch fixtures and exact synthetic-user quota status for the deprecated-email caller. Added four fail-closed controls and successful/rejected-path lookup assertions. Complete suites35/35 plus nearby regressions24/24 pass; Black, Ruff, normal hooks and bounded review pass. Bandit has zero new findings against baseline, excluding intentional test assertions. Production behavior is unchanged; whole-PR review and merge gates remain open.
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
