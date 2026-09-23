---
id: TASK-13330
title: >-
  Consolidate retry backoff into core/Utils/backoff.py by moving it out of
  http_client
status: In Progress
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-23 19:39'
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
- [x] #3 ADR records the jitter algorithm choice
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
APPLIED, with the schedule decision recorded as ADR-047.

DECISION (repository owner, 2026-09-22): decorrelated jitter for outbound HTTP only; in-process contention stays on short capped exponential. One algorithm is NOT imposed on both - HTTP retries cross a network to a shared endpoint where jitter de-synchronises a fleet, while SQLite lock contention is single-process with a millisecond window, and decorrelated jitter prev*3 grows faster than capped exponential so it would have lengthened lock retries to solve a problem those sites do not have.

NEW: core/Utils/backoff.py - decorrelated_jitter_delay, capped_exponential_delay, parse_retry_after_seconds, is_sqlite_locked_error.

MOVED OUT of core/http_client.py rather than copied, so that 6,600-line module SHRINKS: 6678 -> 6656 lines. It re-exports both under their original private names because two test files monkeypatch them; verified the names remain importable and callable.

MIGRATED, all behaviour-preserving (jitter=False where that is the current behaviour):
- transaction_utils.py - 0.1 * 2**n
- Workflows_DB.py - FOUR sites, not the two the finding identified (1668, 1682, 2202, 2307). Also removed a hardcoded tries<4 in _sqlite_retry_commit that ignored its sibling max_tries parameter.

Tests: tests/Utils/test_backoff_schedules.py, 12 cases pinning BOTH schedules against the sequences the existing call sites used - 0.05/0.1/0.2/0.4/0.8 for the SQLite loops, 0.2/0.4/0.8/1.6 for transaction_utils - plus the jitter band, the cap, the case-insensitive locked predicate, and Retry-After in both delta-seconds and HTTP-date forms. One test asserts decorrelated jitter grows FASTER than capped exponential, which is the reason the two stay separate.

MISTAKE WORTH RECORDING: my first Workflows_DB edit corrupted indentation because the 20-space sleep pattern is a SUBSTRING of the 24-space lines, so str.replace matched inside them. Caught by ast.parse in the same command, file restored from git, redone with a line-anchored regex.

Regression: app imports; backoff + lint suites green; 74 passed in the transaction/Workflows set. The one failure there (test_dlq_replay_real_allowed) is identical with and without the change (stash-isolated). tests/http_client has widespread PackageNotFoundError failures in this venv, also identical both ways - the package is not pip-installed here.

STILL OPEN: the 28 inline loops in PromptStudioDatabase.py, deliberately left to the decomposition in TASK-13318 rather than edited in place. The shared helper now exists for them.

2026-09-23 reconciliation:
AC3 met - Docs/ADR/047-retry-backoff-schedules.md (Accepted, commit 865012cf6f) records decorrelated jitter for outbound HTTP and capped exponential for in-process contention, with rejected alternatives.
AC1 NOT met (partial) - core/Utils/backoff.py owns delay computation (decorrelated_jitter_delay, capped_exponential_delay, parse_retry_after_seconds); tests/Utils/test_backoff_schedules.py 12 passed. But HTTP retriability classification did not move: _should_retry and _is_dns_resolution_error (the third function the task said to move) are still defined in core/http_client.py (~:1990, :2318). backoff.py's only classifier is is_sqlite_locked_error, and per ADR-047's 2026-09-23 follow-up contention classification is now owned by core/DB_Management/retry_policy.py (TransientContentionError), not backoff.py.
AC2 NOT met (partial) - http_client imports the delay and Retry-After functions from backoff.py (no local definitions remain), but still defines the retriability classifier. Moving _is_dns_resolution_error/_should_retry (or amending AC1 to 'delay computation' only) closes AC1 and AC2 together.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
