---
id: TASK-251
title: Add VN generation call recovery orchestration
status: Done
assignee: []
created_date: '2026-05-10 23:32'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: provider-call transaction and recovery orchestration for scripted VN generation. Scope is backend service/repository recovery primitives; full interpreter integration remains Task 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate same-key request while provider call is in progress returns generation_request_in_progress and does not call the provider twice
- [x] #2 Same-key reclaim works before provider_call_started_at
- [x] #3 Stale lease after provider start marks request/action abandoned and requires a new key
- [x] #4 Completed request replay returns stored response without a provider call
- [x] #5 Stale scene version returns before provider invocation
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Add focused service-level tests for generation-call idempotency and lease recovery using a fake generation adapter.
2. Add narrow VN Play service orchestration helpers that create/replay generation rows, mark provider start before invocation, persist successful revisions, and replay completed responses.
3. Add stale lease handling that abandons provider-started in-flight requests and blocks same-key retries after abandonment.
4. Keep full script interpreter integration out of scope for this task; expose the helper for Task 5 to call.
5. Verify with focused VN Play tests, compile checks, Bandit on touched code, and `git diff --check`.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Added a narrow `execute_script_generation_call` helper for Task 5 to call when the interpreter reaches a model-backed generate opcode.
- The helper replays completed same-key actions before scene-version validation so a lost HTTP response can be retried after the committed scene has moved on.
- The helper rejects provider-started in-flight same-key calls with `generation_request_in_progress`, and abandons expired provider-started leases with `generation_attempt_abandoned`.
- Full script interpreter event/scene integration remains scoped to Task 5.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Verification
<!-- SECTION:VERIFICATION:BEGIN -->
- `python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py tldw_Server_API/tests/VN_Play/test_vn_play_generated_outputs.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q --tb=short --disable-warnings` -> 71 passed
- `python -m compileall -q tldw_Server_API/app/core/VN_Play tldw_Server_API/tests/VN_Play/test_vn_play_action_requests.py` -> passed
- `python -m bandit -r tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/constants.py -f json -o /tmp/bandit_vn_generation_orchestration.json` -> 0 findings
- `git diff --check` -> passed
<!-- SECTION:VERIFICATION:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added recoverable scripted VN generation call orchestration primitives: idempotent request/action replay, provider-start checkpointing, stale lease abandonment, completed response replay, and strict provider-output revision persistence through the existing generation repository and parser/adapter seam. Full interpreter wiring remains the next task.
<!-- SECTION:FINAL_SUMMARY:END -->
