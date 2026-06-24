---
id: TASK-2407
title: Harden audio core review findings
status: Done
assignee: []
created_date: 2026-06-23 18:11
updated_date: 2026-06-24 03:31
labels:
- audio
- security
- review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated findings from the current-code review of tldw_Server_API/app/core/Audio. Scope includes WebSocket auth token handling, quota fail-open exception classes, TTS WebSocket producer cancellation, tokenizer loader local-only enforcement, tokenizer payload validation, and duplicate audio helper ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings are either fixed with tests or documented as not applicable.
- [x] #2 WebSocket audio auth no longer accepts long-lived credentials in query strings by default.
- [x] #3 Quota fail-open exception tuples do not catch programmer errors such as NameError.
- [x] #4 TTS WebSocket producer cancels promptly when the consumer exits or send fails.
- [x] #5 Tokenizer model loading honors auto_download/local-only policy and rejects malformed payloads with 400-level errors.
- [x] #6 Duplicate helper ownership is resolved or clearly delegated to core implementations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-audio-core-review-hardening.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan: Docs/superpowers/plans/2026-06-23-audio-core-review-hardening.md

Verification results:
- py_compile passed for touched audio source and tests.
- Direct executable checks passed for query-token auth gating, quota exception tuples, TTS producer cancellation and normal queue drain, tokenizer local-only enforcement, malformed tokenizer payload errors, and endpoint BYOK delegation.
- Focused pytest with parent conftest cut off passed: Audio/test_audio_streaming_service_core.py plus fail-open/BYOK tests, 6 passed in 35.20s.
- Focused tokenizer pytest with parent conftest cut off passed: 5 passed in 6.56s.
- Full selected pytest using repository-global conftest was interrupted after 154s because app lifecycle startup/teardown stalled in pytest cleanup; no assertion failures were observed before interruption.
- Bandit on touched audio source scope completed with 0 results and 0 errors (/tmp/bandit_audio_core_2407.json).

Final verification refresh after tokenizer loader-function refinement:
- py_compile passed for updated tokenizer source and tests.
- Focused tokenizer pytest with parent conftest cut off passed: 6 passed in 1.74s.
- Bandit rerun on touched audio source scope completed with 0 results and 0 errors (/tmp/bandit_audio_core_2407.json).

PR review follow-up:
- Added docstrings for new stream/tokenizer helper functions flagged by Qodo.
- Preserved parent cancellation by re-raising asyncio.CancelledError in producer/consumer paths before broad noncritical handlers.
- Added regression tests for parent cancellation during TTS stream cleanup and failed completion-sentinel enqueue cancelling the consumer instead of hanging.
- Focused audio pytest after follow-up: 8 passed in 3.62s.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and fixed all reviewed audio-core findings: disabled WebSocket query-token credentials by default behind AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH, removed NameError from quota fail-open exception tuples, fixed TTS WebSocket producer/consumer cancellation without truncating normal streams, made tokenizer loading fail closed when local-only cannot be enforced, mapped malformed tokenizer inputs to structured HTTP 400 responses, sanitized BYOK user-id logging, and delegated the aggregate endpoint BYOK wrapper to the core TTS helper. Added focused regression coverage for each validated behavior.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR review follow-up 2:
- Verified the completed-consumer observation gap is still present in the producer-done branch of streaming_service.py.
- Applying the minimal fix to observe consumer_task when it is already in the done set.
Follow-up verification:
- py_compile passed for streaming_service.py and test_audio_streaming_service_core.py.
- Focused audio pytest passed: 9 passed in 1.95s.
- git diff --check passed.
- Bandit touched audio scope completed with 0 results and 0 errors (/tmp/bandit_audio_core_2446_observe_consumer.json).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
