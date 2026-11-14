---
id: TASK-15496
title: Own-key calls through analyze skip the DEFAULT_MODEL_<PROVIDER> fallback
status: To Do
labels:
- byok
- llm
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15426 PR B's plan review (B-6); measured by PR D's plan refresh (V11) and its review (2026-11-12). Pre-existing for every analyze call that carries a user's own-key handle. Summarization_General_Lib._summarize_via_adapter takes the model of a call with a provider_credentials handle only from the handle's app_config, the provider section's model or model_id (core/LLM_Calls/Summarization_General_Lib.py:303-311; lines on 7f30182940). A call on the server's key falls back to resolve_provider_model, which also reads the DEFAULT_MODEL_<PROVIDER> environment variable (core/LLM_Calls/adapter_utils.py:160-180). So on a server whose only default model for a provider is that variable (no model for it in config.txt), an own-key analyze call has no model and fails before any provider call. Measured through POST /api/v1/media/process-documents with PR D's prototype, the adapter stubbed, DEFAULT_MODEL_OPENAI=gpt-env-default set and no openai model in config: on the user's own key HTTP 200, the item a Warning with no analysis (Analysis failed for chunk 1: Model is required for provider 'openai'); on the server's key the call reaches gpt-env-default. Affected: PR B's benchmark judge and summaries, PR C's RAG and quality judges, and with PR D every media analysis of a user with a stored key, under or over a limit (PR D adds no media-only guard: controller ruling, 2026-11-13). Reach: the shipped config.txt sets a model for every provider and the real _build_app_config (core/Provider_Credentials/byok_runtime.py:975) copies it into the handle; no Dockerfile, compose file or .env.example sets DEFAULT_MODEL_*, so it needs an operator who removed the config model. Constraint on the fix: it belongs in _summarize_via_adapter, which every handle caller shares. The provider-override model policy is enforced at resolve time against the configured model only (core/Provider_Credentials/provider_credential_runtime.py:1063-1083), and _summarize_via_adapter does not check it again at dispatch, so a fallback model chosen at dispatch has to pass that policy too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An own-key analyze call on a server whose only default model for the provider is DEFAULT_MODEL_<PROVIDER> (no model in config) reaches that model, with a test
- [ ] #2 That model is checked against the provider-override model policy, as the configured model is, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
