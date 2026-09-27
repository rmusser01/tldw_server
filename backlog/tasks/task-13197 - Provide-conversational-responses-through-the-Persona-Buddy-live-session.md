---
id: TASK-13197
title: Provide conversational responses through the Persona Buddy live session
status: Done
assignee: []
created_date: 2026-09-05 21:29
updated_date: 2026-09-06 02:10
labels:
- persona
- buddy
- uat
dependencies: []
references:
- Docs/Reviews/MIGU_BUDDY_MERGED_LIVE_UAT_2026_09_05.md
- TASK-13180
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-merge Migu UAT on dev 220bf544b7 reproduced a usability gap: a simple greeting/request for an exact reply returns a rag_search tool plan and no conversational response. The Buddy transport and review feedback work, but ordinary conversation cannot yet pass provider-response acceptance. Preserve explicit tool approval and the existing Persona Chat/provider ownership boundaries; determine and record the relevant ADR before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A simple conversational prompt through the real Buddy live session returns a correlated provider-backed answer without requiring an unrelated RAG tool approval.
- [x] #2 Stop cancels active generation and a subsequent send can complete in the same session without a late response replacing current feedback.
- [x] #3 Tool-requiring requests retain explicit review and existing authentication, scope and approval policy.
- [x] #4 Real provider UAT and targeted regression evidence identify the exact tested revision and provider configuration without exposing secrets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Authenticated chat reuse and cancellable live turn ownership cross transport/provider boundaries.
1. Specify conversational intent routing through the existing authenticated chat request pipeline, preserving tool approval.
2. Add regression tests for correlated replies, stop/retry, owner/lifecycle checks and unavailable providers.
3. Implement bounded connection-owned turns and reuse canonical chat admission/provider services. Preserve FIFO execution for typed turns on the same connection/session; a new send does not cancel earlier turns. Explicit Stop invalidates active and queued tasks across the owned session, while exact task release preserves siblings. Retire the stopped connection queue so a fresh send can proceed even if cancelled work delays acknowledgement.
4. Run targeted checks and real configured provider UAT; record sanitized evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ordinary persisted Persona Live turns now return provider-backed replies through the complete authenticated Chat HTTP boundary, with bounded enabled Persona context and session history. Only the selected credential family and original proxy provenance are forwarded. Explicit tool intent retains Live plan review; slash commands and empty sends cannot reach Chat command preprocessing. Active and queued turns remain FIFO until explicit Stop retires ownership; late replies cannot publish and retry remains available. ADR046 records the contract. Rebased implementation 2270153980 on dev f6d6a673b6 passed 204 targeted Python and 198 frontend tests, OpenAPI fingerprint, zero touched Bandit findings, and Ruff for new Python helpers/tests. Real DeepSeek UAT returned the expected answer in 0.77 seconds, canceled without late output, recovered in the same session, prepared Whisper/Kokoro and emitted 25388 speech bytes for a synthetic transcript. Explicit search retained an unapproved rag_search plan; REST Stop returned 200. Sanitized evidence identifies the source. Human voice acceptance remains TASK13202. Renumbered from TASK13194 after rebase because dev independently allocated that ID to video Service Prompts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Real DeepSeek response, Stop/retry and explicit tool-review acceptance passed on the rebased implementation. Human speech acceptance remains separately tracked.
<!-- SECTION:FINAL_SUMMARY:END -->

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
