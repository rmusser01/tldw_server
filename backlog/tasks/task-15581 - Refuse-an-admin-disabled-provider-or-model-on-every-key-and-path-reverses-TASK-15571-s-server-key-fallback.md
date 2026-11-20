---
id: TASK-15581
title: Refuse an admin-disabled provider or model on every key and path (reverses
  TASK-15571's server-key fallback)
status: To Do
labels:
- llm
- byok
- policy
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner's rule, 2026-11-19, verbatim: "if a user tries to use a model/service thats been explicitly disabled by the admin, they shouldn't be able to use it; if its just been disabled as in 'no credits/key enabled' then tehy should be able to use it via BYOK". An explicit disable is the provider override's is_enabled=False (provider_disabled), or a model not on its allowed_models when that list is set (model_not_allowed); the owner confirmed the second on 2026-11-19. Either one is refused on every key (the user's own, a team or org shared key, and the server's) and on every path. A provider the server simply has no key or credits for is not a disable: a user's own key must still work. This reverses TASK-15571's rule (a denied model falls back to the server key), which PR #3523 shipped (b1a32fbba4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every provider adapter call (chat, achat, stream, astream) refuses an admin-disabled provider or model on every key (own, team, org, server, no handle) before the quota check, with ChatConfigurationError carrying the reason as error_code, with tests
- [ ] #2 The analyze path (_summarize_via_adapter) refuses with 'configuration' on every key and with no handle, whatever the model's source (configured, DEFAULT_MODEL_<PROVIDER>, model_override); the evaluation judges and Prompt Studio's direct adapter calls refuse a disallowed DEFAULT_MODEL_<PROVIDER>; the MCP model_completion authoritative path refuses on every key after its scope check, with tests
- [ ] #3 An override store that cannot be read refuses (credential_store_unavailable) on each path above, with tests
- [ ] #4 A provider the server has no key or credits for still serves a user's own key: with no override, or one that only leaves it on, an own-key call runs and is not checked against the LLM limit, with a test
- [ ] #5 PR #3523's fallback is removed (server_key_fallback, log_server_key_fallback, override_outcome, the quota gate's strict parameters, the media ingest worker's answered marking, the judges' and MCP's server-key branches) and llm_quota_gate.py, media_ingest_jobs_worker.py, eval_runner.py and ms_g_eval.py are as before #3523
- [ ] #6 ADR-086 is amended with the reversal and the owner's rule; Usage_Quotas.md, the BYOK guide (and its Docs/Published mirror) and Docs/Architecture/llm-providers.md say a disabled provider or model is refused on every key
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
