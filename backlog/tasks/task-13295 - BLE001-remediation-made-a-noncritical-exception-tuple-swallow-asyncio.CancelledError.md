---
id: TASK-13295
title: >-
  BLE001 remediation made a noncritical-exception tuple swallow
  asyncio.CancelledError
status: To Do
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-22 20:05'
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
- [ ] #2 A failing test asserts an over-quota upload returns 413 rather than 200/207
- [x] #3 asyncio.CancelledError is removed from _PERSISTENCE_NONCRITICAL_EXCEPTIONS
- [ ] #4 HTTPException is removed from the tuple, or every suppress site that must not swallow it is narrowed
- [x] #5 The two sibling tuples in Audio/ are corrected in the same pass
- [x] #6 A tests/lint/ AST rule rejects any BaseException-derived member in a *_NONCRITICAL_EXCEPTIONS tuple, seeded so it cannot regress
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch ci/review-followup-visibility. Two of the task's claims did not survive verification; the core defect did, and is larger than filed.

VERIFIED (the real defect): asyncio.CancelledError derives from BaseException, not Exception, so the bare 'except Exception:' these tuples replaced never caught it. Listing it widened what is suppressed. Proved end to end, not just by MRO: test_cancellation_during_ledger_init_propagates drives _get_media_ingestion_daily_ledger, whose 'await ledger.initialize()' sits inside a tuple-guarded try. Against unfixed code the cancellation is logged as 'Media ingestion budget: failed to initialize ResourceDailyLedger:' -- with an empty message, because CancelledError carries none -- and the function returns None, so the caller proceeds as though the ledger were merely unavailable.

SCOPE CORRECTION: the task says 'the two sibling tuples in Audio'. An AST scan of all 282 *_NONCRITICAL_EXCEPTIONS tuples in app/ found 41 files carrying CancelledError, across endpoints, core and services -- admin, auth, chat, notes, sandbox, watchlists, workflows, MCP_unified, TTS, WebSearch, and four background schedulers/aggregators. All 41 are fixed in this pass. Two were missed by the first sweep because they are annotated assignments (moderation_pipeline.py, Workflows/engine.py) -- the ratchet walks ast.AnnAssign as well as ast.Assign, which is how they surfaced.

AC #2 DISPROVEN, not deferred. Both HTTP 413 quota raises are already guarded by an 'except HTTPException: raise' clause placed ahead of the tuple handler -- persistence.py:2977 for the upload path and :5023 for the per-item path. An over-quota upload therefore returns 413 today; it cannot return 200/207 by this mechanism. The finding did not check for the re-raise guard.

AC #4 NOT TAKEN. HTTPException in the tuple is a latent hazard, but it is a *different* class of defect from the BaseException one and the tuple is used at ~140 sites in persistence.py alone. Removing it changes which exceptions escape ~140 handlers, with no failing test to anchor the change now that AC #2's premise is gone. That belongs in its own task with its own reproduction.

AC #6 delivered as tests/lint/test_noncritical_exception_tuples.py: AST rule, zero allowance, rejects BaseException/KeyboardInterrupt/SystemExit/GeneratorExit/CancelledError in any *_NONCRITICAL_EXCEPTIONS tuple. It carries a companion test asserting the premise on the live interpreter, so if the language ever changes the rule fails loudly instead of silently arguing with it. Red before the sweep listing all 41 files; green after.

Verification: new tests 18 passed (ratchet + persistence behaviour + the TASK-13303 OCR file). ruff on all 41 changed app files shows no new codes -- the 113 reported are the repo's existing non-blocking baseline categories (UP045/I001/B023/F821), and specifically zero unused-asyncio-import findings. tests/lint/test_endpoint_auth_deps_import_boundary.py fails, but it also fails on a clean tree (verified by stashing) -- pre-existing, unrelated. Bandit not installed in this environment, so DoD #4 is a documented skip.

KNOCK-ON EFFECTS OF THE SWEEP (found by running the suites for every touched module, then diffing against a clean-dev baseline):

1. REGRESSION, found and fixed: tests/sandbox/test_ws_connection_quotas.py::test_sandbox_ws_per_user_quota_enforced_and_released passed 3/3 on origin/dev and failed 3/3 with the sweep -- deterministic, not flake. Cause: sandbox.py's WS finally released the quota slot AFTER 'await stream.ws.close()'. An await inside a finally that is already unwinding a cancellation re-raises CancelledError immediately, so the release was abandoned and the slot leaked for the run's lifetime; the second connect returned close code 4429. The suppression was what had been hiding it. Fix: release the slot before the socket teardown -- everything above it is synchronous, so the release no longer depends on how the socket goes away. 3/3 green after.

That is the honest shape of this whole task: the cleanup was riding on the bug. Restoring the suppression would have hidden it again.

2. Systematic follow-up rather than one-off: an AST scan of all 41 changed files for the same shape -- a finally whose first await is followed by more statements -- found 19 sites. Triaged, 15 are best-effort (logging, metrics, socket close, idempotent done-flags) and 3 already guard cancellation explicitly ('except asyncio.CancelledError: raise' or 'suppress(asyncio.CancelledError, Exception)'). One, audio_streaming.py:1817, does strand a release_context_connection() -- but its preceding await is handled by EXPECTED_DB_EXC, a DB tuple that never contained CancelledError, so it is pre-existing on dev and not caused by this change. Left alone, noted here.

3. MCP_unified/server.py: deregistration reordered ahead of 'await stream.stop()' on the same reasoning. Stated honestly -- I wrote a test (tests/MCP_unified/test_mcp_ws_connection_release_on_cancel.py) expecting it to reproduce a leak and it passed with BOTH orderings, because an ordinary disconnect arrives as WebSocketDisconnect and the finally runs to completion; the cancelled path was not reachable from TestClient. The reorder is kept as free hardening, not as a fix for a reproduced defect, and both the code comment and the test docstring say so. The test is retained for its own sake: it is the first coverage asserting that registry and the per-IP counter drain at all.

4. A fourth candidate in Audio_Streaming_Unified.py was changed and then reverted, for the same reason -- no failing case could be constructed, and unlike the MCP reorder the change added nesting and re-introduced local CancelledError suppression, which is the opposite of this task's point.
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
