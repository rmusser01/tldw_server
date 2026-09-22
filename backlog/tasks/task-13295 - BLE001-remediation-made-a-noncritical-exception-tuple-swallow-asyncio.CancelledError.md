---
id: TASK-13295
title: >-
  BLE001 remediation made a noncritical-exception tuple swallow
  asyncio.CancelledError
status: In Progress
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-22 05:08'
labels:
  - bug
  - ingestion
  - reliability
dependencies: []
references:
  - 'tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py:83'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/Ingestion_Media_Processing/persistence.py:83-105` defines `_PERSISTENCE_NONCRITICAL_EXCEPTIONS` with **`asyncio.CancelledError` as its first entry**.

Verified:
```
issubclass(asyncio.CancelledError, Exception) -> False
CancelledError.__mro__ -> [CancelledError, BaseException, object]
```

So the `except Exception:` this tuple was written to replace **would not have caught cancellation** — it propagated correctly. The explicit tuple written to satisfy ruff BLE001 now catches it. **The lint remediation introduced a defect the lint rule cannot see.**

The tuple is used across 12 `contextlib.suppress` sites (`:872, 1005, 2118, 2236, 3971, 3979, 4172, 5087, 5090, 5093, 5096, 5721`), several wrapping awaits. A client disconnect mid-`POST /media/add` is absorbed: the coroutine keeps writing to the media DB for a client that is gone, and `Task.cancel()` never completes, so graceful shutdown blocks.

The same tuple contains `HTTPException` while the file raises it 11 times — including the 413 over-quota raise, which is then swallowed, so an over-quota upload can return 200/207 instead of 413.

Sibling tuples with the same defect, currently inert: `Audio/Audio_Streaming_Unified.py:80`, `Audio/Audio_Transcription_Lib.py:93`.

This is the inverse of the review noise floor: reviewers were told not to re-report grandfathered BLE001 files, and the interesting defect turned out to be in the *remediation*.

Found by the comprehensive core-module review; the MRO and subclass relationship independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failing test cancels a task mid-persistence and asserts CancelledError propagates rather than being suppressed
- [ ] #2 A failing test asserts an over-quota upload returns 413 rather than 200/207
- [ ] #3 asyncio.CancelledError is removed from _PERSISTENCE_NONCRITICAL_EXCEPTIONS
- [ ] #4 HTTPException is removed from the tuple, or every suppress site that must not swallow it is narrowed
- [ ] #5 The two sibling tuples in Audio/ are corrected in the same pass
- [ ] #6 A tests/lint/ AST rule rejects any BaseException-derived member in a *_NONCRITICAL_EXCEPTIONS tuple, seeded so it cannot regress
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SCOPE CORRECTION during Stage 0: the defect is systemic, not 3 files. An AST sweep found asyncio.CancelledError in 39 *_NONCRITICAL_EXCEPTIONS tuples across 39 files (19 under api/v1/endpoints, 20 under core). 16 of those files ALREADY contain an explicit "except asyncio.CancelledError: raise" handler, proving the authors knew it must propagate and that the tuple membership is a mistake.

Done: all 39 removed; new AST ratchet at tldw_Server_API/tests/lint/test_noncritical_exception_tuples.py (red before, green after) bans CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit and BaseException from any *_NONCRITICAL_EXCEPTIONS tuple. All 39 files parse and app.main imports.

HTTPException: NOT removed from the tuple - 130 sites catch _PERSISTENCE_NONCRITICAL_EXCEPTIONS and a wholesale removal is not a zero-risk change. Instead the specific 413 path was fixed surgically with "except HTTPException: raise" immediately before the tuple catch (persistence.py:5078), matching the file own idiom at :2815, :2975, :3041.

asyncio.TimeoutError was checked and deliberately LEFT - it is an alias of the builtin TimeoutError, an Exception subclass, so it is safe.
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
