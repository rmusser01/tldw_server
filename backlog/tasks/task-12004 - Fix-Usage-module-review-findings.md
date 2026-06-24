---
id: TASK-12004
title: Fix Usage module review findings
status: Done
assignee: []
created_date: '2026-06-23 20:27'
updated_date: '2026-06-23 20:47'
labels:
  - usage
  - quota
  - budget
  - review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated current-code review findings in `tldw_Server_API/app/core/Usage`: audio minute ledger correctness, ledger-unavailable quota behavior, single-operation daily-minute consumption, cancellation propagation, USD budget safety for placeholder pricing, metrics cardinality, and Usage module README accuracy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated same-duration audio minute events are counted separately in ResourceDailyLedger.
- [x] #2 Ledger-unavailable audio minute quota paths preserve enforcement semantics or surface quota-store failure for bounded fail-open handling.
- [x] #3 Daily audio minute check/consume has a single operation that prevents stale separate check/add endpoint usage.
- [x] #4 Placeholder or unknown billable model prices cannot silently produce zero USD budget usage.
- [x] #5 Audio quota helpers do not swallow asyncio.CancelledError.
- [x] #6 Usage metrics avoid per-user high-cardinality labels.
- [x] #7 Usage README reflects current ledger/RG/AuthNZ behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-usage-module-review-fixes.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added focused regression tests in `tldw_Server_API/tests/Usage/test_usage_review_fixes.py`.
- Added unique/default audio minute operation ids and explicit operation-id support for idempotent callers.
- Added `ResourceDailyLedger.consume_if_available()` and `audio_quota.consume_daily_minutes()` for store-backed daily-minute enforcement plus recording.
- Updated audio transcription and streaming call sites to use consume semantics while retaining legacy test-shim compatibility.
- Removed per-user Prometheus LLM metrics; durable AuthNZ usage logs remain the per-user source of truth.
- Changed placeholder pricing to use conservative billable fallback rates while preserving documented free non-placeholder zero rates.
- Updated `tldw_Server_API/app/core/Usage/README.md` for ledger/RG/AuthNZ behavior and documented the `audio_quota.py` boundary to discourage adding more unrelated responsibilities before a dedicated extraction.
<!-- SECTION:NOTES:END -->

## Modified Files

<!-- SECTION:MODIFIED_FILES:BEGIN -->
- `Docs/superpowers/plans/2026-06-23-usage-module-review-fixes.md`
- `tldw_Server_API/app/api/v1/endpoints/audio/__init__.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`
- `tldw_Server_API/app/core/DB_Management/Resource_Daily_Ledger.py`
- `tldw_Server_API/app/core/Usage/README.md`
- `tldw_Server_API/app/core/Usage/audio_quota.py`
- `tldw_Server_API/app/core/Usage/pricing_catalog.py`
- `tldw_Server_API/app/core/Usage/usage_tracker.py`
- `tldw_Server_API/tests/Usage/test_usage_review_fixes.py`
<!-- SECTION:MODIFIED_FILES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- Red test confirmation: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Usage/test_usage_review_fixes.py` initially failed 8 tests as expected.
- Focused Usage suite: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Usage` -> 57 passed, 1 skipped.
- Relevant Audio endpoint/quota suite: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py tldw_Server_API/tests/Audio/test_audio_transcription_language_normalization.py tldw_Server_API/tests/Audio/test_audio_transcriptions_timed_segments.py tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py tldw_Server_API/tests/Audio/test_audio_router_import_resilience.py tldw_Server_API/tests/Audio/test_audio_streaming_truthiness_flags.py tldw_Server_API/tests/Audio/test_audio_quota_rg_and_ledger.py tldw_Server_API/tests/Audio/test_audio_quota_unit.py` -> 117 passed.
- Syntax: `source .venv/bin/activate && python -m compileall -q ...` on touched Python paths -> passed.
- Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Usage tldw_Server_API/app/core/DB_Management/Resource_Daily_Ledger.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py tldw_Server_API/app/api/v1/endpoints/audio/audio.py tldw_Server_API/app/api/v1/endpoints/audio/__init__.py -f json -o /tmp/bandit_usage_12004.json` -> 0 findings, `results: []`.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Usage review findings by making audio minute usage ledger entries unique by default, adding a store-backed consume operation for daily-minute enforcement, surfacing ledger-unavailable failures to bounded fail-open paths, preserving cancellation propagation, removing per-user Prometheus LLM labels, pricing billable placeholders conservatively, and refreshing Usage documentation.
<!-- SECTION:FINAL_SUMMARY:END -->
