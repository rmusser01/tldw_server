---
id: TASK-13299
title: >-
  Evaluations Unix timestamps are wrong by the host UTC offset, violating
  ADR-014
status: Done
assignee: []
created_date: '2026-09-22 04:51'
updated_date: '2026-09-23 00:12'
labels:
  - bug
  - evaluations
  - database
dependencies: []
references:
  - 'tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:2489'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/DB_Management/Evaluations_DB.py:_ensure_unix_timestamp` converts stored timestamps to Unix seconds:

```python
dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
...
return int(dt.timestamp())
```

Against SQLite's own `CURRENT_TIMESTAMP` format (`2026-09-21 20:00:15`) all three steps fail:

1. `.replace("Z","+00:00")` is a **no-op** — that format has no `Z`.
2. `fromisoformat` returns a **naive** datetime (`tzinfo=None`).
3. `.timestamp()` on a naive datetime interprets it in the **host local zone**.

Executed under `TZ=America/Los_Angeles`:
```
naive .timestamp() : 1790046015
true UTC           : 1790020815
delta              : 25200 s = exactly 7 h
```

The `except` fallback on the same function returns `int(datetime.now().timestamp())` — also naive, same class of error.

**ADR-014 names Unix `created` timestamps as a preserved OpenAI-compatible convention**, so this is a binding-contract violation, not just a bug. Every evaluation `created` value is wrong by the host offset on any non-UTC deployment — which for a self-hosted product is the common case.

Three further copies carry the identical defect: `unified_evaluation_service.py:1503`, `evaluations_datasets.py:51`, `evaluations_rag_pipeline.py:72,116,174`. On PostgreSQL the datasets converter matches no `isinstance` branch and falls through to `now()`, so a dataset's `created` becomes the time it was **read** — the same field, two different wrong answers split by backend.

**No test catches it because CI runs UTC, where the delta is exactly zero.**

Found by the comprehensive core-module review; independently reproduced by the orchestrator under a non-UTC TZ.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test under a non-UTC TZ asserts the Unix timestamp matches true UTC
- [x] #2 All four conversion sites treat naive stored timestamps as UTC
- [x] #3 The except fallback no longer uses naive datetime.now()
- [x] #4 The PostgreSQL datasets path no longer falls through to now() for an unmatched type
- [x] #5 One CI shard runs under a non-UTC TZ so this class of defect is visible -- this is the durable fix
- [x] #6 Conformance with ADR-014's Unix created convention is asserted by test
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed test-first and merged to dev in PR #2980 (merge commit 8045fa2956). A failing test reproduced the defect before any code changed, with controls pinning the behaviour that had to stay unchanged. Qodo review then found follow-on defects in three of this batch's fixes; those were corrected in the same PR before merge.
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
