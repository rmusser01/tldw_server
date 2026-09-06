---
id: TASK-13207
title: Address Persona UAT PR review diagnostics and contracts
status: Done
created_date: 2026-09-06 03:43
references:
- https://github.com/rmusser01/tldw_server/pull/2908#issuecomment-5556684127
- TASK-13202
updated_date: 2026-09-06 03:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve concrete Qodo review findings for PR #2908 while preserving the requester-deferred physical voice and Whisper responsiveness follow-up in TASK-13202.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Voice commit policy lookup does not execute synchronous database work on the socket event loop.
- [x] #2 Unexpected turn failures produce safe server diagnostics without exposing provider payloads or credentials.
- [x] #3 Conversation context assembly resides in core; new exceptions, docstrings and test contracts follow repository conventions.
- [x] #4 Review dispositions and targeted verification are recorded; deferred voice qualification remains explicitly open.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Review fixes preserve the existing authenticated Chat, context bounds and voice ownership contracts.
1. Add failing socket regressions for nonblocking policy reads and safe failure diagnostics.
2. Move ordinary conversation profile/context assembly into the existing core module and centralize exceptions without changing exception behavior.
3. Complete public docstrings, categorize/type new tests and split auth scenarios; retain focused boundary unit coverage with a documented review disposition.
4. Run targeted tests, lint and Bandit; respond to each Qodo finding and retain TASK-13202 as the authorized deferred voice work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Qodo findings 1–10 addressed: profile loading and bounded context construction delegate to core complete_persona_turn; endpoint retains transport publication and exact owner/session revalidation. Voice commit policy lookup runs through asyncio.to_thread. Unexpected task failures log safe session/correlation and error class, never provider exception values or locals, then release ownership and send the existing terminal notice. Custom exceptions moved to core/exceptions.py with compatibility imports. Public contracts and registry methods documented. Four new test modules are typed and categorized; unauthorized HTTP admission is independently tested. STT configuration/PCM/transcript helpers moved to public core/Persona/live_stt.py so focused selection tests no longer assert private endpoint tuples. Qodo finding 11 matches the already requester-deferred Whisper responsiveness issue and remains open under TASK-13202; no inference performance fix or final physical voice acceptance is claimed. Three new regressions first failed for the expected missing core boundary, blocking lookup and missing diagnostics; 66 focused tests then passed and 218 broader Persona tests passed (4 warnings). Ruff/Black focused scope clean; endpoint retains only its pre-existing SIM114 suggestion. Touched Python Bandit zero findings. No new ADR: preserves ADR046 contracts; no frontend production changes after the 309-test/TypeScript validation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved Qodo's concrete diagnostics, layering, exception and test-contract findings without changing provider admission or voice semantics. The requester-deferred whole-turn Whisper responsiveness and physical floating-state UAT remain tracked in TASK-13202.
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
