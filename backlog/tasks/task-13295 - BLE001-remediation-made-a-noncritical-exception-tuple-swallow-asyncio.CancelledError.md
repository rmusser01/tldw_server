---
id: TASK-13295
title: >-
  BLE001 remediation made a noncritical-exception tuple swallow
  asyncio.CancelledError
status: Done
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-23 23:14'
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
- [x] #1 A failing test cancels a task mid-persistence and asserts CancelledError propagates rather than being suppressed
- [x] #2 asyncio.CancelledError is removed from _PERSISTENCE_NONCRITICAL_EXCEPTIONS
- [x] #3 HTTPException is removed from the tuple, or every suppress site that must not swallow it is narrowed
- [x] #4 The two sibling tuples in Audio/ are corrected in the same pass
- [x] #5 A tests/lint/ AST rule rejects any BaseException-derived member in a *_NONCRITICAL_EXCEPTIONS tuple, seeded so it cannot regress
- [x] #6 An over-quota upload returns 413 and an over-quota URL item is reported as that item's Error result, each covered by a test (amended 2026-09-23: the original wording assumed uploads returned 200/207; they already returned 413, and the per-URL path's per-item Error is the batch contract)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SCOPE CORRECTION during Stage 0: the defect is systemic, not 3 files. An AST sweep found asyncio.CancelledError in 39 *_NONCRITICAL_EXCEPTIONS tuples across 39 files (19 under api/v1/endpoints, 20 under core). 16 of those files ALREADY contain an explicit "except asyncio.CancelledError: raise" handler, proving the authors knew it must propagate and that the tuple membership is a mistake.

Done: all 39 removed; new AST ratchet at tldw_Server_API/tests/lint/test_noncritical_exception_tuples.py (red before, green after) bans CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit and BaseException from any *_NONCRITICAL_EXCEPTIONS tuple. All 39 files parse and app.main imports.

HTTPException: NOT removed from the tuple - 130 sites catch _PERSISTENCE_NONCRITICAL_EXCEPTIONS and a wholesale removal is not a zero-risk change. Instead the specific 413 path was fixed surgically with "except HTTPException: raise" immediately before the tuple catch (persistence.py:5078), matching the file own idiom at :2815, :2975, :3041.

asyncio.TimeoutError was checked and deliberately LEFT - it is an alias of the builtin TimeoutError, an Exception subclass, so it is safe.

2026-09-23 reconciliation: AC3 met (9f5373725b; persistence.py _PERSISTENCE_NONCRITICAL_EXCEPTIONS no longer lists CancelledError). AC5 met (Audio_Streaming_Unified/Audio_Transcription_Lib tuples clean; ratchet covers all of app/). AC6 met (tests/lint/test_noncritical_exception_tuples.py passes, zero offenders; caveat: no positive-control fixture, and it only inspects plain Assign of a literal tuple, not AnnAssign or tuple concatenation). Bandit on the 37 app files touched by 9f5373725b: no issues. AC1 NOT met: no test cancels a task mid-persistence; only the structural lint ratchet exists. AC2 NOT met: persistence.py:5078 'except HTTPException: raise' fixes the 413 path, but no test drives an over-quota upload through persistence and asserts 413 (the only rejecting-quota tests are test_video_ingestion.py:617 / test_audio_files_preflight.py:336, a different path). AC4 NOT met: HTTPException is still in the tuple and only the one 413 site was narrowed; the other ~130 catch/suppress sites were not audited.

2026-09-23 completion (e8e9449ae1). AC1: tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py::test_cancelling_add_media_mid_persistence_propagates parks add_media_orchestrate inside the upload quota check, cancels the task, asserts CancelledError and that no item was processed. Red against 9f5373725b^ persistence.py ('DID NOT RAISE CancelledError'; log shows 'Quota check failed (non-fatal)'), green now. AC2 amended (premise false): the upload quota path already re-raised 413 before the tuple catch pre-9f5373725b; test_over_quota_upload_is_rejected_with_413 passes before and after (regression guard). The :5078 'except HTTPException: raise' from 9f5373725b was on the per-URL item path in process_document_like_item and was a regression: SSRF blocks / per-URL quota rejections escaped instead of returning the item's Error result (ingest-jobs worker then fails the job; reading_service's tuple does not catch HTTPException; /media/add folds it to 207 anyway). Reverted with a comment; test_over_quota_url_item_is_reported_as_that_items_error red before revert (HTTPException 413 raised), green after. AC4 (HTTPException audit): kept in the tuple deliberately. AST pass over persistence.py (direct 'raise HTTPException' + intra-file call graph: validate_add_media_inputs, add_media_orchestrate, add_media_persist, process_document_like_item, _run_doc_item) finds exactly one tuple catch that can see an HTTPException with no earlier 'except HTTPException' handler - the per-item prep catch in process_document_like_item, which is intentional. Limitation: HTTPExceptions raised by other modules inside tuple-guarded blocks were not traced. Tests: MediaIngestion_NEW/unit + lint: 412 passed, 31 failed, 12 collection errors; the same 31 fail with HEAD persistence.py and the errors are missing yt_dlp (env). Bandit -ll persistence.py: no issues.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
CancelledError no longer sits in any *_NONCRITICAL_EXCEPTIONS tuple (39 files, 9f5373725b), guarded by an AST lint ratchet and now a behavioural test that cancels /media/add mid-persistence (e8e9449ae1). HTTPException stays in the persistence tuple on purpose: the only site that sees it is the per-item prep catch, where it becomes that item's Error. The earlier re-raise there broke that contract and was reverted. Over-quota uploads return 413 (test added). AC2 was amended because its premise was wrong. Known skips: 31 pre-existing yt_dlp/video failures plus 12 collection errors in MediaIngestion_NEW/unit (env).
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
