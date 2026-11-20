---
id: TASK-15576
title: Decide whether the resolve-time provider-override runtimes follow TASK-15571's
  server-key fallback
status: In Progress
labels:
- llm
- byok
- policy
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15571's plan review (I5). TASK-15571 applies the owner's rule (a model the provider-override policy denies falls back to the server key) at analyze dispatch (_summarize_via_adapter), in users_own_key_credentials, and on the MCP authoritative path. The 19 other runtimes built with override_snapshot_resolver still refuse a denied model at resolve time (ProviderCredentialRuntime._capture_override_snapshot / _enforce_call_policy). That covers chat, character chat, chat documents, messages, audio streaming, TTS and realtime, RAG unified, evaluations unified, Prompt Studio and its jobs worker, chunking, moderation, the auto boundary assistant and speech chat. Chat also answers 403 per request (validate_provider_override, chat.py:4468). Some of their handles reach _summarize_via_adapter, for example RAG HyDE (hyde.py:132), so a denied configured model still has two outcomes by endpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether the server-key fallback extends to these runtimes and to chat's 403
- [ ] #2 If it does: each runtime family falls back with the same fail-closed quota check, with tests; if not: the docs say which endpoints refuse
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the final review of PR #3523: hyde.py:132-150 marks the user's handle used after sgl.analyze returns, even when the call ran on the server key (a denied-model fallback), so only the key's last-used time is wrong. Any change to the 19 runtimes inherits it. Also for the owner's decision: server-key calls are never checked against the override, so a disabled provider stops BYOK handle calls but not server-key calls on these paths (plan Decision 2).
From PR #3523's final review (2026-11-19): (1) the study assistants also refuse a denied model on every key: resolve_chat_target calls validate_provider_override (core/Chat/chat_target_resolution.py, from API_Deps/study_assistant_deps.py and core/Flashcards/study_assistant.py) and answers 'Provider configuration is unavailable.', as chat does; the docs say so. (2) 'still refuse a denied model at resolve time' holds for the configured model only: Prompt Studio's direct-adapter helpers (evaluation_manager._call_adapter_text, prompt_executor.py, test_runner.py) take model or resolve_provider_model(provider, handle.app_config), and the resolve-time _enforce_call_policy checks only the configured model (configured_provider_model_from_snapshot ignores DEFAULT_MODEL_*), so a denied DEFAULT_MODEL_<PROVIDER> can still run there on a user or team key. Same bug class as TASK-15571's dispatch-time case; decide it with the rest.
Owner's answer, 2026-11-19: keep refusing; explicit disable is refused everywhere. So no runtime adopts the fallback, and #3523's is removed (TASK-15581). Chat's 403 and the 19 resolve-time runtimes are unchanged; the provider base class now refuses at dispatch too, which closes the Prompt Studio DEFAULT_MODEL_* gap, and the HyDE mark-used note goes away with the fallback (a call either runs on the handle or raises).
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
