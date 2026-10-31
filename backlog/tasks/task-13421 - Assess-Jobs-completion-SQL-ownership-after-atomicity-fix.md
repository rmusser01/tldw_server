---
id: TASK-13421
title: Assess Jobs completion SQL ownership after atomicity fix
status: To Do
created_date: 2026-10-03 01:33
references:
- TASK-13215
- https://github.com/rmusser01/tldw_server/pull/3092#discussion_r4171067327
documentation:
- Docs/ADR/058-jobs-completion-row-identity.md
- Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review Qodo PR #3092 finding dd3ff113-e31c-4ca9-a49d-48c73a4273eb against the existing Jobs facade and the repository DB_Management-only SQL guidance. The focused TASK-13215 lock/UUID fix intentionally preserves the current transaction owner and excludes extraction. Evaluate a separately designed boundary for completion SQL without splitting the authoritative lock, identity guard, RLS context, counters or outbox transaction. This task does not authorize broad migration or behavioral fixes during strict extraction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inventory existing completion SQL ownership and approved strict-extraction boundaries before proposing changes.
- [ ] #2 Propose an explicit transaction/RLS-preserving ownership decision with risks and verification gates, for separate human approval.
- [ ] #3 Keep behavior changes, caller migration and historical repair separate from extraction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
