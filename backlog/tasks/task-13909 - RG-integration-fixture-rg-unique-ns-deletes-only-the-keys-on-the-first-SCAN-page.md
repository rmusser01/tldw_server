---
id: TASK-13909
title: RG integration fixture rg_unique_ns deletes only the keys on the first SCAN
  page
status: To Do
labels:
- resource-governance
- redis
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/Resource_Governance/integration/conftest.py cleans a test's namespace with one client.scan(0, match=..., count=1000) call, before and after the test, and never follows the cursor. On a Redis with more than about 1000 keys the namespace's keys are often on a later page, so they stay. Found while planning TASK-13882/TASK-13892 (2026-10-22): the shared local Redis held about 1150 keys, 520 of them rg_it_* keys from past runs (328 namespaces, no TTL). CI's Redis is ephemeral, so this only fills shared developer Redis instances.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 rg_unique_ns deletes every key in its namespace whatever the keyspace size (follow the SCAN cursor, for example with scan_iter), with a check against a Redis holding more than 1000 other keys
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
