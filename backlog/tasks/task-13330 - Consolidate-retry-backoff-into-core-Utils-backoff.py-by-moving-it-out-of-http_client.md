---
id: TASK-13330
title: >-
  Consolidate retry backoff into core/Utils/backoff.py by moving it out of
  http_client
status: To Do
assignee: []
created_date: '2026-09-22 04:58'
labels:
  - duplication
  - utils
  - migration
dependencies: []
references:
  - 'tldw_Server_API/app/core/http_client.py:2311'
  - 'tldw_Server_API/app/core/RAG/rag_service/resilience.py:281'
  - 'tldw_Server_API/app/core/DB_Management/transaction_utils.py:68'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eight independent module-level backoff implementations plus 28 inline loops. Ranking the three adopted ones by reading them:
- http_client._decorrelated_jitter_sleep (2311-2315): min(cap, uniform(base, prev*3)) - genuine DECORRELATED jitter, the algorithm that de-synchronises a retrying fleet - paired with delta-seconds AND HTTP-date Retry-After parsing (2318-2337) and a classifier treating DNS failures as permanent (2340-2357). BEST.
- resilience.RetryPolicy._calculate_delay (281-289): symmetric +/-25% jitter, which keeps clients clustered in a narrow band; no Retry-After, no classifier. Middling.
- transaction_utils.py:68: 0.1 * (2 ** retry_count), ZERO jitter. Worst.

EXPLICIT VERDICT: do NOT promote resilience.py - that would standardise on the weaker algorithm and entrench a RetryPolicy name collision that already exists inside core/RAG/ (two different classes with incompatible constructors).

Destination: core/Utils/backoff.py, SEEDED BY MOVING the three http_client functions OUT of it. This satisfies the constraint against growing the 6,600-LOC junk drawer by actively shrinking it. Jitter algorithm is an operational decision, so it needs an ADR.

Source: synthesis F30
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 core/Utils/backoff.py owns delay computation and retriability classification
- [ ] #2 http_client imports from it rather than defining it
- [ ] #3 ADR records the jitter algorithm choice
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
