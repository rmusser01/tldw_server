---
id: TASK-13302
title: Evaluations created timestamps violate ADR-014 by the host UTC offset
status: Done
assignee: []
created_date: '2026-09-22 04:52'
updated_date: '2026-09-23 19:47'
labels:
  - bug
  - evaluations
  - adr-drift
dependencies: []
references:
  - 'tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:2489'
  - 'tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py:1503'
  - 'tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py:51'
  - Docs/ADR/014-evaluations-openai-compatible-schemas.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_ensure_unix_timestamp does datetime.fromisoformat(s.replace("Z","+00:00")) then int(dt.timestamp()). Verified under TZ=America/Los_Angeles against SQLite CURRENT_TIMESTAMP format:
- .replace("Z","+00:00") is a no-op, that format has no Z
- fromisoformat returns a NAIVE datetime
- .timestamp() interprets it in the host local zone: 1790050015 vs true UTC 1790024815, delta 25200s = exactly 7h

ADR-014:12 names Unix created timestamps as a preserved OpenAI-compatible convention, so this is a binding-contract violation. The except fallback returns int(datetime.now().timestamp()), also naive. Three further copies carry the identical defect. On PostgreSQL the datasets converter matches no isinstance branch and falls through to now(), so every dataset created becomes the time it was READ.

No test catches it because CI runs UTC where the delta is exactly zero. See companion task for the non-UTC CI shard.

Source: synthesis F6
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Naive SQLite timestamps are interpreted as UTC, not local
- [x] #2 Fallback path no longer substitutes a naive now()
- [x] #3 PostgreSQL datetime inputs are handled rather than falling through to now()
- [x] #4 Test asserts a fixed stored value converts identically under two TZ settings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE (converter). Evaluations_DB._ensure_unix_timestamp now treats naive datetimes as UTC via a local _to_epoch helper, and all five naive datetime.now() fallbacks became datetime.now(timezone.utc). Fixes both the string-parse path AND the isinstance(value, datetime) branch at :2483, which had the same defect and was not in the original finding.

KEY CHOICE: rather than depending on a CI timezone change, the test CARRIES ITS OWN TZ - tldw_Server_API/tests/Evaluations/unit/test_created_timestamp_utc_contract.py sets TZ=America/Los_Angeles with time.tzset() and restores it. So it catches this class even in UTC CI, which removes the hard dependency on TASK-13336. Red before (2 failed on the naive cases), green after (4 passed).

Verified no regression: tests/Evaluations/unit went from 27 failed to 25 failed with the change applied - my fix turned 2 red into green and broke nothing. The remaining 25 are pre-existing ModuleNotFoundError: sklearn failures (sklearn is a DECLARED core dep at pyproject.toml:87 but missing from this venv).

STILL OPEN: the three bypassing copies (unified_evaluation_service.py:1503, evaluations_datasets.py:51, evaluations_rag_pipeline.py:72/116/174) and the PostgreSQL fall-through-to-now() path. Those are owner-only for the two endpoint files.

2026-09-23 reconciliation: AC2 met - Evaluations_DB._ensure_unix_timestamp fallbacks use _utc_now_epoch() (datetime.now(timezone.utc)); datasets endpoint fallback also already aware. (Premise note: naive datetime.now().timestamp() is in fact the correct epoch, so this fallback was never numerically wrong; the change is hygiene.) AC4 met - tests/Evaluations/unit/test_created_timestamp_utc_contract.py (commit 7c348a05ae) forces TZ=America/Los_Angeles via tzset and asserts identical conversion under LA and UTC: 4 passed. AC1 PARTIAL, NOT checked - fixed in Evaluations_DB (verified naive datetime and naive string both map to UTC epoch under TZ=America/Los_Angeles), and datasets/runs/evals all route through it so the datasets endpoint's own string branch is unreachable in practice. But evaluations_rag_pipeline.py to_ts() copies at :72-79, :116-123, :173-180 still call .timestamp() on naive fromisoformat/strptime results, so pipeline preset created_at/updated_at are still off by the host UTC offset. unified_evaluation_service._extract_created_ts (:1496) has the same defect but has no callers (dead code - delete). AC3 NOT checked - for datasets the premise is weaker than stated: rows reach _normalize_dataset_payload already carrying an int 'created' from _row_to_dataset_dict, whose isinstance(datetime) branch now treats naive as UTC, so PG datetimes are handled. But the pipeline-preset to_ts() does '"T" in x' on a PG datetime -> TypeError -> created_at=None, and _normalize_dataset_payload still has no datetime branch. Remaining: route the three rag_pipeline to_ts copies (and the datasets endpoint branch) through the DB converter, delete _extract_created_ts. Bandit on Evaluations_DB.py: no findings.

2026-09-23: AC1+AC3 done (a250262292). Evaluations_DB.to_unix_timestamp is now module-level (method delegates); the three evaluations_rag_pipeline to_ts() copies and the datasets normalizer's string branch route through it, so naive SQLite strings and PostgreSQL datetimes both read as UTC. Deleted uncalled UnifiedEvaluationService._extract_created_ts. Tests: test_created_timestamp_utc_contract.py +2 (PG naive datetime; dataset payload for str and datetime under TZ=America/Los_Angeles) -> 6 passed; the dataset test fails on HEAD. tests/Evaluations: 18 failures before and after, identical set. Bandit (uvx, -ll): no findings. Docs: none needed (ADR-014 already states the contract). No known skips.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All Evaluations created/updated timestamps go through one UTC-aware converter; naive SQLite strings and PostgreSQL datetimes are read as UTC under any host TZ, tested under America/Los_Angeles.
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
