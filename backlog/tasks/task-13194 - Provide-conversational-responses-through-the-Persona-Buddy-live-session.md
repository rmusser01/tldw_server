---
id: TASK-13194
title: Provide conversational responses through the Persona Buddy live session
status: In Progress
created_date: 2026-09-05 21:29
labels:
- persona
- buddy
- uat
priority: high
references:
- Docs/Reviews/MIGU_BUDDY_MERGED_LIVE_UAT_2026_09_05.md
- TASK-13180
updated_date: 2026-09-05 23:18
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
Credential-binding review follow-up: extracted _persona_conversation_headers at the existing WebSocket auth seam. Internal Chat forwarding now includes only the credential family selected by server-owned persona_auth_method: API key, AuthNZ/MCP JWT, or authenticated single-user cookie plus its CSRF token. Competing credentials and ambient cookies are omitted. Original proxy headers remain ordered (including repeated forwarding values) for the same trusted-proxy calculation when the original ws.client is supplied to ASGI; no master credential or bypass marker is introduced. ADR-046 applies.

Tests added in tldw_Server_API/tests/Persona/test_live_conversation_credentials.py. Corrected baseline projection: 13 failed, 2 passed; helper implementation: 15 passed. Targeted command: source /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/activate; python -m pytest tldw_Server_API/tests/Persona/test_live_conversation_credentials.py tldw_Server_API/tests/Persona/test_persona_ws.py -k 'credential or auth or scope' -q --tb=short => 28 passed, 77 deselected, 7 existing warnings (8.81s). New tests Ruff check/format pass. Endpoint Bandit using server2 project venv: zero findings before/after. git diff --check passes. Whole-endpoint Ruff has its existing SIM114 plus temporary F841 for the still-unused conversation_headers placeholder until the separately authorized conversational callsite is integrated. P2 publication guards were already applied by the main task and were not duplicated. No external calls, microphone use, or git mutations performed.
FIFO compatibility correction: same-connection/session typed turns now serialize using per-session asyncio locks, while the receive loop remains responsive. The ownership registry retains active and queued tasks; only explicit Stop invalidates all owned tasks, and exact release preserves siblings. Cancel/REST Stop retires the queue lock so fresh sends are not blocked by retired work delaying cancellation. Updated ADR-046 and the existing implementation plan to describe FIFO and explicit Stop semantics. Added registry ownership/release and queued-Stop regression coverage; the existing client-message correlation test is unchanged. Red evidence: 2 registry failures and 2 WebSocket failures before the correction. Green: pytest test_live_conversation.py test_persona_ws.py test_persona_live_control_api.py: 147 passed, 41 warnings (73.87s); final strengthened queue checks: 2 passed, 89 deselected (2.15s). Scoped Ruff introduces no findings; production retains 3 existing/pending findings, including conversation_headers F841 pending the root-owned conversational integration. Scoped Bandit: 0 before and after. git diff --check passed. Changes remain uncommitted; task remains In Progress for root-owned integration and UAT.
Target resolution now leaves effective credentials to the authenticated Chat HTTP route, avoiding a false rejection of user/team/org BYOK-only text requests. Voice preparation retains a separate conservative server-credential check and documents that BYOK voice is not yet qualified. Core helper suite: 15 passed. Conversational callsite is still intentionally unapplied: automatic approval review rejected forwarding Persona history/memory/context to the configured provider without sufficiently explicit payload/destination authorization. User question is pending; two new conversation WebSocket acceptance tests remain red until this integration is authorized and implemented. No real provider request was sent in the server readiness probe.
Final combined implemented scope: 198 passed, 4 warnings (93.09s), /private/tmp/migu-server-final-targeted.log; includes core conversation, credential binding, full WebSocket/control and new voice runtime suites. The separate two intentionally-red conversation integration tests remain outside this passing scope because the callsite is awaiting the pending payload authorization. Review also found Chat slash-command preprocessing could execute /skill or /weather even without request tools. The adapter now rejects slash commands before HTTP admission with actionable Live-review guidance; both regressions failed before the guard and pass afterward. ADR046 records this boundary. Touched production Bandit: zero findings; new conversation/registry/test Ruff scope clean; full endpoint retains known style debt and the intentional pending unused conversation_headers variable. Latest server dev fetched at f6d6a673b6; rebase still to be performed after completing pending integration.
The user explicitly authorized the full bounded Persona payload to the configured Chat provider, including DeepSeek for UAT. Integration is applied: ordinary persisted Live turns use the authenticated Chat HTTP pipeline with enabled context/history; explicit tool requests retain Live plans. Real provider probe returned the exact expected reply in 1.11s, canceled generation without publishing its answer, recovered in the same session, produced 25388 bytes of Kokoro output for a supplied voice_commit transcript, and retained an unapproved rag_search plan. Real browser setup test also returned the expected provider reply. Scope is provider/TTS, not human microphone/STT/playback acceptance. Integrated 201 tests pass; final command-gate regression rejects empty sends and validates the actual bounded outgoing message, closing a reviewed slash-command bypass (3 red before fix, conversation scope23 green). Evidence: Docs/Reviews/MIGU_BUDDY_MERGED_LIVE_UAT_2026_09_05.md. Task remains In Progress until final rebased PR checks; TASK13195 tracks human voice acceptance.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
